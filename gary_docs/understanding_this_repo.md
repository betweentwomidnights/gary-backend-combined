
# Understanding This Repo

This repo has a few configurations to support two backends. Everything here is designed around this function in the audiocraft repo:

```python
def generate_continuation(self, prompt: torch.Tensor, prompt_sample_rate: int,
                              descriptions: tp.Optional[tp.List[tp.Optional[str]]] = None,
                              progress: bool = False, return_tokens: bool = False)             -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
        """Generate samples conditioned on audio prompts.

        Args:
            prompt (torch.Tensor): A batch of waveforms used for continuation.
                Prompt should be [B, C, T], or [C, T] if only one sample is generated.
            prompt_sample_rate (int): Sampling rate of the given audio waveforms.
            descriptions (list of str, optional): A list of strings used as text conditioning. Defaults to None.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if prompt.dim() == 2:
            prompt = prompt[None]
        if prompt.dim() != 3:
            raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
        prompt = convert_audio(prompt, prompt_sample_rate, self.sample_rate, self.audio_channels)
        if descriptions is None:
            descriptions = [None] * len(prompt)
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, prompt)
        assert prompt_tokens is not None
        tokens = self._generate_tokens(attributes, prompt_tokens, progress)
        if return_tokens:
            return self.generate_audio(tokens), tokens
        return self.generate_audio(tokens)
```

This function works globally now with all the audiocraft models, as the newest version of the repo has a `genmodel.py` that is shared among them.

**Note:**
- **Magnet** kinda sucks at continuations, but it might be used to do some neat sample generation on very simple input audio.
- **Audiogen** takes too long to be enjoyable as a gary.

---

This repo uses docker-compose.

- `Dockerfile.redis` is the image we use for the buffer of 256mb: `thecollabagepatch/redis:latest`

This is supposed to get around the error:

``` 
'the packet is too large' with the input audio. 
```

**IT DIDN'T**. This is one of those things someone can figure out. Right now, you can only continue like twice. I crop a lot, so it's hard to tell.

- `concurrent_gary` is the backend for the chrome extension/web app.
- `g4lwebsockets` is the backend for gary4live, the max4live device.

---

The `old_random_backup_scripts` folder has a bunch of trash cuz I'm a hoarder. A lot of those scripts don't work.

There are some docker-compose files in there, though, that are for combining both backends into a multi-gpu setup. The relevant files for that setup have the word `yikes` in them.