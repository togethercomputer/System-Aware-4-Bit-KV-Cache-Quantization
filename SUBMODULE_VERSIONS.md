# Pinned submodule versions

These are the commits referenced by this repository’s `git submodule` entries. Update this file when you bump submodules for a paper revision.

| Submodule | Branch | Commit |
|-----------|--------|--------|
| [third_party/sglang-fast-rotation](third_party/sglang-fast-rotation) | `jinda_rotation_fast` | `11cdba6681e48d213d630b2e9cbd49774df3eee7` |
| [third_party/sglang-kmeans](third_party/sglang-kmeans) | `jinda_kmeans_rotation_dump` | `43925c00fb91ce58eb2d9c6836bb2f9885ff618f` |

Both submodules use the upstream fork: [github.com/jindajia/sglang-fork](https://github.com/jindajia/sglang-fork).

## Nested submodule (`tore-eval`)

[third_party/sglang-fast-rotation/tore-eval](third_party/sglang-fast-rotation/.gitmodules) points at a private Together repository. If `git submodule update --init --recursive` fails, skip nested init and use [OpenAI simple-evals](https://github.com/openai/simple-evals) from this repo’s docs instead, or substitute your own evaluation harness against SGLang’s OpenAI-compatible HTTP API.
