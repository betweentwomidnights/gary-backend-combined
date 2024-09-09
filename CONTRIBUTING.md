
# contributing to gary

this file is a random explosion of thoughts right now.


# thoughts

we need a faster musicgen model. alot of ppl want a higher quality one, but i think a much faster base model is what's really needed to make this as fun as it can be.

a CLI along with the train-gary module is prolly coming. google keeps bonking me from doing multi-gpu training setups, which the train-gary docker-compose will definitely want.

https://github.com/betweentwomidnights/train-gary has early drafts of fine-tuning scripts based on lyra's, with demucs and some other things added.

if you want to help me build this stuff, i mostly need help with organizing. 

the beginnings of stable audio open gary is in https://github.com/betweentwomidnights/latent-mixer-m4l 


this repo is all up inside the audiocraft repo. i think that's prolly the wrong way to do it.


# Contributing to AudioCraft

We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests

AudioCraft is the implementation of a research paper.
Therefore, we do not plan on accepting many pull requests for new features.
We certainly welcome them for bug fixes.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to encodec, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
