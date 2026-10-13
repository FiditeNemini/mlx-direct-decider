# Project name and earlier naming research

[Back to home](../README.md)

The project is **MLX ZeroEffort Decider**, hosted at
[ItsOdeLeo/MLX-ZeroEffort-Decider](https://github.com/ItsOdeLeo/MLX-ZeroEffort-Decider).
The owner selected this name on **2026-10-13**. Its tagline is:
**Decisions from existing models. No fine-tuning required.**

| Public name | Value |
| :--- | :--- |
| Distribution and CLI | `mlx-zeroeffort-decider` |
| Python import | `mlx_zeroeffort_decider` |
| Local model variable | `MLX_ZEROEFFORT_MODEL` |
| Opt-in model-test variable | `MLX_ZEROEFFORT_TEST_MODEL` |

The `jev_mlx` import, `jev-mlx` command, `JEV_MLX_MODEL` and `JEV_TEST_MODEL`
remain compatible. The canonical and legacy Python namespaces share the same
classes and implementation; requests and sessions can be used through either
namespace. No model weights, inference behavior, or cache policy changed.
The implementation source stays under `src/jev_mlx` so existing source links
and callers remain valid. A rename does not publish a package to PyPI.

Here, ZeroEffort refers to the absence of an additional fine-tuning workflow.
Applications still supply a compatible local checkpoint, state and candidates.

## Implementation and compatibility

The new namespace delegates to the existing implementation so request/result
classes keep their identity across imports. Package metadata and the shared CLI
register the new command and model variable; the legacy command and variable
remain accepted, with an explicit `--model` taking precedence. These connections
belong to the packaging/parser rather than a model adapter. Optional-backend
installation errors now name the new distribution. They do not affect scoring.

Server health metadata and current UI labels use the new project name; clients
that compare the literal health `project` string should use the new name.
Benchmark environment metadata records the new distribution version. These are
identity/diagnostic changes, with no inference, scheduling or cache changes.
Historical benchmark files are preserved. Namespace identity, environment
precedence, optional-dependency-free imports and both CLI entries are verified
separately from the recorded model-quality evaluations.

## Previous names and recorded evidence

Earlier names include **JEV MLX**, **JEVKit MLX** and **MLXJ**. The display-only
rename from MLXJ to JEV MLX occurred on 2026-09-20. Recordings, frozen protocols,
and benchmark evidence retain their original names, commands, source paths and
bytes. Older reproduction commands still work through compatibility entries.
The GitHub repository keeps its existing history and issues under the new name.

This remains an independent project inspired by Jev, not an official TypeSafe AI
release or the existing [bnsd55/jevmlx](https://github.com/bnsd55/jevmlx) project.
It does not use Jev weights or reproduce unpublished training methods.

## Earlier research

Research date: 2026-09-19. The table below preserves the earlier public-search snapshot, not the current naming decision, a reservation, or a guarantee that a registry will accept a name.

| Candidate | Findings at the time | Assessment |
| --- | --- | --- |
| JEV MLX / jevmlx | [bnsd55/jevmlx](https://github.com/bnsd55/jevmlx) already exists in the same field | Similar names exist; identify this project by its owner and repository URL |
| JF | [jf on PyPI](https://pypi.org/project/jf/) is already a JSONL query tool | Too short and loses the MLX identity |
| JF MLX | GitHub repository-name searches for `jf-mlx` / `jfmlx` returned no matches | The specific remembered project was not identified; this does not prove uniqueness |
| MLXJ | GitHub `mlxj in:name` returned partial matches, with no exact `mlxj` repository basename found; the [MLXJ account](https://github.com/MLXJ) already exists | Earlier display name; replaced by JEV MLX to make the inspiration explicit |

At the time, the [PyPI mlxj JSON API](https://pypi.org/pypi/mlxj/json) returned 404; the corresponding `jf-mlx` and `jev-mlx` APIs also returned 404. This only means no public project record was found, not that registration is guaranteed. No account, repository, or package was created or published.

GitHub repository-name queries can be repeated through the [official repository search API](https://api.github.com/search/repositories?q=mlxj+in%3Aname). Search results can change.

## README references

The page structure was informed by these first-party READMEs, without copying their graphics, branding, or prose:

- [LobeHub](https://github.com/lobehub/lobehub#readme): centered visual title, language choice, community and support links.
- [RAGFlow](https://github.com/infiniflow/ragflow#readme): separate language READMEs, documentation navigation, and clear sections.
- [Clash Verge Rev](https://github.com/clash-verge-rev/clash-verge-rev#readme): identity, language entry points, and sponsorship information.

Language is chosen on the homepage: the default English homepage is `README.md`, and the Chinese homepage is `README.zh-CN.md`. Each leads to a complete set of documents in that language, with no language switch on individual documents. Relative links work on GitHub without JavaScript. New visual assets are original project SVGs.

Public social links use owner-provided account details. On 2026-10-08 the owner updated the display name to **欧的Leo** and both the Xiaohongshu ID and X handle to **ItsOdeLeo**. The README and follow badges use [@ItsOdeLeo on X](https://x.com/ItsOdeLeo) and the existing [Xiaohongshu profile link](https://xhslink.com/m/18bjTTf180W). The old profile card is no longer displayed on the homepage because its printed account details are outdated; [social provenance](assets/social/README.md) retains the source history. A sponsorship destination remains unconfigured.
