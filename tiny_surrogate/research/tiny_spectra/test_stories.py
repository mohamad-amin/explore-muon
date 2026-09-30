"""Data-only independence and no-recycling contracts; no model scores."""
import unittest
import numpy as np
import torch
from .stories import (TokenCorpus, read_complete_stories, document_identity,
                      assign_exact, quarantine_overlap, evaluation_starts,
                      evaluation_plan_all_targets,padded_evaluation_batch)


def docs(texts,source):
    return [dict(source=source,source_index=i,text=text,text_sha256="unused",identity=document_identity(text))
            for i,text in enumerate(texts)]


class StoryDataContracts(unittest.TestCase):
    def test_trim_incomplete_utf8_story_before_decode(self):
        rows,trim=read_complete_stories(b"One.<|endoftext|>\nTwo.<|endoftext|>\npartial\xe2","train")
        self.assertEqual([r["text"] for r in rows],["One.","Two."])
        self.assertEqual(trim["complete_document_count"],2)
        self.assertGreater(trim["dropped_incomplete_suffix_bytes"],0)

    def test_normalized_identity_precedes_role_and_dedup(self):
        self.assertEqual(document_identity(" HELLO, world! "),document_identity("hello WORLD"))
        train,candidates,audit=assign_exact(docs(["HELLO world","fresh train"],"train"),
                                           docs(["hello, world!","held out story","HELD OUT STORY"],"valid"))
        self.assertEqual(len(train),2)
        self.assertEqual(len(candidates),1)
        self.assertEqual(audit[2]["reason"],"exact_normalized_duplicate_of_train")
        self.assertEqual(audit[4]["reason"],"exact_normalized_duplicate_of_valid")

    def overlap(self,traintexts,validtexts,roles=None,hash_function=None):
        train,candidates,audit=assign_exact(docs(traintexts,"train"),docs(validtexts,"valid"))
        if roles:
            for d,role in zip(candidates,roles):d["role"]=role
        kwargs={} if hash_function is None else dict(hash_function=hash_function)
        kept,evidence=quarantine_overlap(train,candidates,audit,**kwargs)
        return kept,evidence

    def test_63_words_do_not_trigger_but64_do(self):
        shared=" ".join(f"w{i}" for i in range(64))
        kept,_=self.overlap([shared+" train"],[" ".join(shared.split()[:63])+" val"])
        self.assertEqual(len(kept),1)
        kept,events=self.overlap([shared+" train"],[shared+" val"])
        self.assertEqual(len(kept),0)
        self.assertEqual(len(events),1)

    def test_all_matching_documents_quarantined(self):
        shared=" ".join(f"w{i}" for i in range(65))
        kept,events=self.overlap([shared+" train"],[shared+" first",shared+" second"])
        self.assertEqual(len(kept),0)
        self.assertEqual(len(events),2)

    def test_hash_collision_requires_exact_words(self):
        def collisions(words,n,cache):
            return ((0,i) for i in range(len(words)-n+1))
        a=" ".join(f"a{i}" for i in range(70))
        b=" ".join(f"b{i}" for i in range(70))
        kept,events=self.overlap([a],[b],hash_function=collisions)
        self.assertEqual(len(kept),1)
        self.assertEqual(events,[])

    def test_no_artificial_cross_document_shingles(self):
        left=" ".join(f"a{i}" for i in range(40));right=" ".join(f"b{i}" for i in range(40))
        kept,_=self.overlap([left,right],[left+" "+right])
        self.assertEqual(len(kept),1)

    def test_dev_test_overlap_quarantines_both_sides(self):
        shared=" ".join(f"w{i}" for i in range(64))
        kept,events=self.overlap(["some training story"],[shared+" dev",shared+" test"],roles=["dev","test"])
        self.assertEqual(kept,[])
        self.assertEqual(events[0]["reason"],"shared64_words_across_dev_test")

    def corpus(self):
        offsets=[dict(start=0,end=13),dict(start=13,end=26)]
        return TokenCorpus({"train":np.arange(81),"val":np.arange(26)},tuple(map(str,range(100))),
                           dict(seq_len=4,default_stream_seed=99),evaluation_starts(offsets,4),dev_documents=offsets)

    def test_fresh_prefix_pairing_and_partition(self):
        a,b=self.corpus(),self.corpus()
        ga,gb=torch.Generator().manual_seed(7),torch.Generator().manual_seed(7)
        small=torch.cat([a.window_positions("train",3,4,generator=ga) for _ in range(4)])
        big=b.window_positions("train",12,4,generator=gb)
        self.assertTrue(torch.equal(small,big))
        self.assertEqual(len(set(big.tolist())),12)
        targets=[i for start in big.tolist() for i in range(start+1,start+5)]
        self.assertEqual(len(set(targets)),len(targets))
        x,y=b.batch("train",12,4,positions=big)
        self.assertTrue(torch.equal(x[:,1:],y[:,:-1]))

    def test_stream_exhaustion_and_no_test_route(self):
        corpus=self.corpus();g=torch.Generator().manual_seed(3)
        corpus.window_positions("train",20,4,generator=g)
        with self.assertRaisesRegex(ValueError,"exhausted"):
            corpus.window_positions("train",1,4,generator=g)
        with self.assertRaisesRegex(ValueError,"Sealed test"):
            corpus.tokens("test")
        with self.assertRaisesRegex(ValueError,"Sealed test"):
            corpus.window_positions("test",1,4,generator=g)

    def test_development_windows_respect_story_and_mapping(self):
        corpus=self.corpus()
        self.assertEqual(corpus.evaluation_starts("val",4).tolist(),[0,4,8,13,17,21])
        self.assertEqual(corpus.evaluation_document_indices("val",4).tolist(),[0,0,0,1,1,1])

    def test_explicit_positions_cannot_bypass_boundary_or_frozen_context(self):
        corpus=self.corpus()
        with self.assertRaisesRegex(ValueError,"document-contained"):
            corpus.batch("val",1,4,positions=[11])
        with self.assertRaisesRegex(ValueError,"Context length"):
            corpus.batch("val",1,3,positions=[0])
        with self.assertRaisesRegex(ValueError,"canonical"):
            corpus.batch("train",1,4,positions=[1])

    def test_v2_all_original_targets_once_including_short_and_tail(self):
        cursor=0;documents=[];expected=[]
        for size in (0,1,2,4,5,6,8,9,10):
            documents.append(dict(start=cursor,end=cursor+size))
            expected.extend(range(cursor+1,cursor+size))
            cursor+=size
        starts,lengths,indices=evaluation_plan_all_targets(documents,4)
        actual=[]
        for start,length,i in zip(starts,lengths,indices):
            actual.extend(range(int(start)+1,int(start+length)+1))
            self.assertGreaterEqual(start,documents[i]["start"])
            self.assertLess(start+length,documents[i]["end"])
        self.assertEqual(actual,expected)
        self.assertEqual(len(actual),len(set(actual)))
        self.assertTrue((lengths>=1).all() and (lengths<=4).all())

    def test_v2_padding_masks_only_artificial_positions_and_keeps_real_eot(self):
        ids=torch.tensor([5,0,8,9,0])
        before=ids.clone()
        x,y=padded_evaluation_batch(ids,[0,2],[1,2],4,eot_id=0)
        self.assertEqual(x.tolist(),[[5,0,0,0],[8,9,0,0]])
        self.assertEqual(y.tolist(),[[0,-100,-100,-100],[9,0,-100,-100]])
        self.assertEqual(int((y!=-100).sum()),3)
        self.assertTrue(torch.equal(ids,before))
        with self.assertRaisesRegex(ValueError,"Negative source"):
            padded_evaluation_batch(torch.tensor([5,-1]),[0],[1],4,eot_id=0)

    def test_v2_corpus_masked_api_and_document_identity(self):
        offsets=[dict(start=0,end=3),dict(start=3,end=10)]
        starts,lengths,indices=evaluation_plan_all_targets(offsets,4)
        corpus=TokenCorpus({"train":np.arange(81),"val":np.arange(10)},tuple(map(str,range(100))),
                           dict(seq_len=4,default_stream_seed=99,eot_id=0,evaluation_policy_version=2),
                           starts,dev_documents=offsets,dev_lengths=lengths,dev_document_indices=indices)
        self.assertEqual(corpus.evaluation_starts("val",4).tolist(),[0,3,7])
        self.assertEqual(corpus.evaluation_lengths("val",4).tolist(),[2,4,2])
        self.assertEqual(corpus.evaluation_document_indices("val",4).tolist(),[0,1,1])
        x,y=corpus.evaluation_batch("val",torch.tensor([0,7]),4)
        self.assertEqual(y.tolist(),[[1,2,-100,-100],[8,9,-100,-100]])
        x2,y2=corpus.batch("val",2,4,positions=[0,7])
        self.assertTrue(torch.equal(x,x2) and torch.equal(y,y2))
        with self.assertRaisesRegex(ValueError,"outside declared"):
            corpus.evaluation_batch("val",torch.tensor([2]),4)


if __name__=="__main__":unittest.main()
