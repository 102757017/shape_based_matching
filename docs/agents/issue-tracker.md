# Issue tracker（问题追踪器）：GitHub

本仓库的 issue 与 spec 都以 GitHub issues 形式存在。所有操作使用 `gh` CLI。

## 约定

- **创建 issue**：`gh issue create --title "..." --body "..."`。多行正文用 heredoc。
- **读取 issue**：`gh issue view <number> --json number,title,body,labels,comments`。
- **列出 issue**：`gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`，配合 `--label` 与 `--state` 过滤。
- **把 issue 设为某父 issue 的子项**：`gh issue create --parent <parent> ...`，或事后 `gh issue edit <parent> --add-sub-issue <child>`（`gh` 2.94+）。旧版 `gh`：用 `gh api --method POST repos/<owner>/<repo>/issues/<parent>/sub_issues -F sub_issue_id=<child-db-id>`（数据库 id，见下文「阻塞」）。若无子 issue 功能，在子 issue 正文顶部写 `Part of #<parent>`。
- **阻塞关系（Blocking）**：采用 GitHub 原生的 issue 依赖，这是 UI 可见的规范表示。用 `gh issue create --blocked-by <n>,<n> ...` 或 `gh issue edit <child> --add-blocked-by <blocker>`（`gh` 2.94+）添加边。旧版 `gh`：用 `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`，其中 `<blocker-db-id>` 是阻塞者的数值 **database id**（`gh api repos/<owner>/<repo>/issues/<n> --jq .id`，注意不是 `#number` 或 `node_id`）。若依赖功能不可用，退而在子 issue 正文顶部写 `Blocked by: #<n>, #<n>`。所有 blocker 关闭后该工单才算解锁。
- **统计未关闭的 blocker**：`gh api repos/<owner>/<repo>/issues/<n> --jq .issue_dependencies_summary.blocked_by`（只算未关闭的；0 表示可开工）。`gh issue view <n> --json blockedBy` 也列出已关闭的 blocker，需自行过滤：`--jq '[.blockedBy.nodes[] | select(.state == "OPEN")] | length'`。
- **评论 issue**：`gh issue comment <number> --body "..."`
- **打 / 去标签**：`gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **关闭**：先 `gh issue comment <number> --body "..."` 说明原因，再 `gh issue close <number>`。对已关闭的 issue，`gh issue close --comment` 仍会丢弃评论并以退出码 0 结束。

仓库地址从 `git remote -v` 推断；在 clone 内运行 `gh` 会自动识别。

只在用户自己的仓库里建 issue。不要建到 `mattpocock/skills`（这些技能的来源仓库），除非用户拥有它。若是 fork，`gh` 可能默认指向上游父仓库——每次会话首次 `gh issue create` 前，先用 `gh repo view --json nameWithOwner` 确认目标，若不是用户自己的仓库则需与用户确认。

## 把 PR 当作 triage 入口

**PR 作为 triage 入口：否。** _（若本仓库把外部 PR 视为功能请求，改为 `yes`；`/triage` 会读取该标志。）_

当为 `yes` 时，PR 与 issue 走相同的标签与状态，使用 `gh pr` 系列命令：

- **读取 PR**：`gh pr view <number>`，再 `gh pr view <number> --comments`（非 TTY 时 `--comments` 只打印评论，不含正文），`gh pr diff <number>` 看差异。
- **列出待 triage 的外部 PR**：`gh api --paginate 'repos/{owner}/{repo}/pulls?state=open' --jq '.[] | select(.author_association | IN("OWNER","MEMBER","COLLABORATOR") | not) | {number, title, author: .user.login, author_association, labels: [.labels[].name]}'`。
- **评论 / 打标签 / 关闭**：`gh pr comment`、`gh pr edit --add-label`/`--remove-label`、`gh pr close`。

GitHub 的 issue 与 PR 共用一套编号，裸 `#42` 可能是任一；用 `gh pr view 42` 解析，失败则退回 `gh issue view 42`。

## 当技能说「发布到 issue tracker」

创建一个 GitHub issue。

## 当技能说「获取相关工单」

按上文「读取 issue」的方式读取。

## 路径导航操作（wayfinding）

供 `/wayfinder` 使用。地图（map）是一个带子 issue 的单一 issue。

- **地图（Map）**：一个带 `wayfinder:map` 标签的 issue，承载 Notes / Decisions-so-far / Fog 正文。创建方式：`gh issue create --label wayfinder:map`。
- **子工单（Child ticket）**：通过 GitHub 子 issue 关联到地图（见上文「把 issue 设为某父 issue 的子项」）。若无子 issue 功能，在地图正文任务列表里加入子项，并在子 issue 正文顶部写 `Part of #<map>`。标签：`wayfinder:<type>`（`research`/`prototype`/`grilling`/`task`）。被认领后指派给负责开发。
- **阻塞**：见上文「阻塞关系」。
- **前沿查询（Frontier query）**：列出地图的未关闭子项（作用域为地图的子 issue / 任务列表），剔除仍有未关闭 blocker 或已被指派者；地图顺序优先者胜出。
- **认领（Claim）**：`gh issue edit <n> --add-assignee @me`，即会话的首次写入。
- **解决（Resolve）**：`gh issue comment <n> --body "<answer>"`，然后 `gh issue close <n>`，再把上下文指针（gist + 链接）追加到地图的 Decisions-so-far。
