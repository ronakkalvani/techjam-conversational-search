# AI Assistance and Runtime Disclosure

## Development assistance

ChatGPT was used during development as an engineering assistant for:

- brainstorming and comparing retrieval and questioning strategies;
- drafting and reviewing implementation changes;
- debugging and test-case design;
- analyzing evaluation results; and
- editing technical and submission documentation.

The team owned the problem framing and final design decisions, reviewed and
integrated the code, and ran the evaluations. The lexical-questioning idea was
originated by Aniket Khan and was inspired by Akinator's question-driven
approach to narrowing a hidden answer.

## Runtime

The submitted baseline does **not** call ChatGPT or any other LLM. It uses only
Python standard-library code, the organizer-provided frozen catalog, and
session text supplied through the official Agent interface. It makes no network
requests, requires no credentials, downloads no model, and reports zero model
tokens.

The repository contains an offline sparse-vector implementation and an
intent-conditioned policy for controlled ablation experiments. Both are
disabled in the final baseline configuration.
