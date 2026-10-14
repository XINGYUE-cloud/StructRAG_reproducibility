# GitHub Pages: A Practical Student Site

GitHub Pages turns a repository into a static website. For a student portfolio, begin with one public repository named YOUR-GITHUB-ID.github.io; its default address is https://YOUR-GITHUB-ID.github.io/.

## Create It

1. Create a **public** repository with the exact name YOUR-GITHUB-ID.github.io.
2. Add an index.html to the repository root. Start with plain HTML; it is easier to maintain than a framework.
3. Push to the main branch.
4. Open **Settings > Pages**, set the deployment source to **Deploy from a branch**, then choose main and / (root).
5. Wait for the deployment status to become successful, then visit the site URL shown in that page.
6. Add a custom domain only after the default site is stable. A custom domain is optional.

For a project-specific page instead, enable Pages on that project repository. Its URL will normally be https://YOUR-GITHUB-ID.github.io/REPOSITORY-NAME/.

## What a Second-Year Software Engineering Student Should Publish

Keep the site compact, factual, and easy to scan:

- A short introduction: year, major, current technical interests, and one sentence about what you are building or learning.
- Three to five selected projects. For each: problem, your contribution, technical choices, result or lesson, source code link, and a screenshot or diagram where useful.
- A research or learning section: paper replications, technical reading notes, benchmark results, and what did not work.
- Skills shown through evidence. Link a technology to the project where you used it rather than listing every tool you have seen.
- Contact links such as GitHub and a professional email address.

For this project, a strong portfolio entry would state: “Reproducing a controlled comparison of text, table, triple, and graph representations for person-profile RAG; documenting retrieval recall and answer accuracy by reasoning type.” Link this repository and a short experiment report.

## Why It Is Worth Having

- It gives reviewers one stable place to inspect your code, documentation, and progress.
- It demonstrates engineering habits: version control, reproducibility, concise writing, and honest experiment reporting.
- It makes internship or lab applications easier because each claim can point to concrete work.
- It compounds over time: short reading notes and small finished projects are more credible together than an empty skills list.

## What Not to Publish

Do not expose student ID numbers, phone numbers, home address, private chat records, API keys, model checkpoints, passwords, or other people's data. Do not claim a paper is reproduced when only a baseline or partial pipeline has been run.
