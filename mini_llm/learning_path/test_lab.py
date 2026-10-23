"""Behavioral checks for the learning path; use unittest, not external test packages."""
import copy
import json
import math
import unittest
import urllib.error
import urllib.request

import torch

import lab
from local_service import serve_local


class ModelChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.model = lab.SinoLM(lab.Config(context_length=16, embedding_dim=12,
                                          num_heads=3, num_layers=1)).eval()

    def test_roundtrip_and_unsupported_input(self):
        self.assertEqual(lab.decode(lab.encode("the cat\n")), "the cat\n")
        with self.assertRaises(ValueError):
            lab.encode("안녕")

    def test_causality_and_broken_mask_detection(self):
        a = torch.tensor([lab.encode("the cat")])
        b = a.clone(); b[:, 4:] = (b[:, 4:] + 11) % len(lab.CHARS)
        torch.testing.assert_close(self.model(a)[0][:, :4], self.model(b)[0][:, :4])
        for block in self.model.blocks:
            block.attention.mask.fill_(True)
        self.assertGreater(float((self.model(a)[0][:, :4] - self.model(b)[0][:, :4]).abs().max().detach()), 1e-5)

    def test_invalid_config_context_and_labels(self):
        with self.assertRaises(ValueError): lab.Config(embedding_dim=12, num_heads=5)
        for length in (0, 17):
            with self.assertRaises(ValueError): self.model(torch.zeros(1, length, dtype=torch.long))
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 2, dtype=torch.long), torch.full((1, 2), -100))

    def test_generation_boundaries_and_no_weight_update(self):
        before = copy.deepcopy(self.model.state_dict())
        self.model.train()
        output = self.model.generate("the ", max_new_tokens=20)
        self.assertEqual(len(output), 24)
        self.assertTrue(self.model.training)
        for name, value in before.items():
            self.assertTrue(torch.equal(value, self.model.state_dict()[name]))
        for options in ({"temperature": 0}, {"temperature": float("nan")},
                        {"top_k": 0}, {"max_new_tokens": -1}):
            with self.assertRaises(ValueError): self.model.generate("a", **options)

    def test_learning_and_finite_gradients(self):
        data = lab.text_ids(lab.sentence_splits()["train"])
        before = lab.evaluate(self.model, data)
        losses = lab.train_lm(self.model, data, steps=30)
        self.assertTrue(all(math.isfinite(value) for value in losses))
        self.assertLess(lab.evaluate(self.model, data), before)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_unique_disjoint_sentence_splits(self):
        splits = lab.sentence_splits()
        self.assertEqual([len(splits[k]) for k in ("train", "val", "test")], [60, 10, 10])
        self.assertEqual(len(set(sum(splits.values(), []))), 80)

    def test_sft_target_shift_and_padding(self):
        x, y = lab.sft_batch(["cat"], 48)
        prompt_length = len(lab.chat_prompt("cat"))
        self.assertTrue((y[0, :prompt_length-1] == -100).all())
        self.assertEqual(lab.decode(y[0][y[0] != -100]), "cat~")
        self.assertEqual(int(y[0, prompt_length-1]), lab.STOI["c"])
        with self.assertRaises(ValueError): lab.sft_batch(["cat"], 8)

    def test_lora_preserves_base_then_updates_only_adapters(self):
        # SFT needs a longer window than the small model used in other checks.
        base = lab.SinoLM(lab.Config(context_length=48, embedding_dim=12, num_heads=3, num_layers=1)).eval()
        adapted = lab.add_lora(base)
        x, _ = lab.sft_batch(["cat", "dog"], 48)
        with torch.no_grad(): torch.testing.assert_close(base(x)[0], adapted(x)[0])
        before = {name: p.detach().clone() for name, p in adapted.named_parameters()}
        lab.fit_sft(adapted, ["cat", "dog"], steps=3)
        changes = {name: not torch.equal(value, dict(adapted.named_parameters())[name]) for name, value in before.items()}
        self.assertTrue(any(changes[name] for name, p in adapted.named_parameters() if p.requires_grad))
        self.assertFalse(any(changes[name] for name, p in adapted.named_parameters() if not p.requires_grad))


class ApplicationChecks(unittest.TestCase):
    def test_retrieval_match_and_abstention(self):
        self.assertEqual(lab.retrieve("borrow books fourteen days")[1]["id"], "borrowing")
        self.assertIsNone(lab.retrieve("volcano spacecraft")[1])
        self.assertEqual(lab.answer_request({"question": "volcano spacecraft"})["status"], "no_match")

    def test_invalid_tool_arguments(self):
        self.assertEqual(lab.answer_request({"tool": "sum", "numbers": [12, 8, 5]})["answer"], 25)
        for payload in ([], {"tool": "shell"}, {"tool": "sum", "numbers": [True]},
                        {"tool": "sum", "numbers": [float("nan")]},
                        {"tool": "sum", "numbers": [1e30]},
                        {"tool": "sum", "numbers": [10**1000]}, {"question": ""}):
            with self.assertRaises(ValueError): lab.answer_request(payload)

    def test_actual_http_boundary_and_shutdown(self):
        def send(url, body, path="/ask"):
            request = urllib.request.Request(url+path, data=body,
                                             headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=3) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as error:
                return error.code, json.load(error)
        with serve_local() as url:
            self.assertEqual(send(url, b'{"question":"opens weekdays"}')[1]["source"], "hours")
            self.assertEqual(send(url, b'broken')[0], 400)
            self.assertEqual(send(url, b'[]')[0], 400)
            self.assertEqual(send(url, b'{}', path="/unknown")[0], 404)
            self.assertEqual(send(url, b'x'*4097)[0], 413)
        with self.assertRaises(urllib.error.URLError): send(url, b'{}')


if __name__ == "__main__":
    unittest.main(verbosity=2)
