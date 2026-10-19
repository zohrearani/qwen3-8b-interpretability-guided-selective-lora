# Publishing this package to GitHub

1. Sign up at https://github.com/signup and confirm your email. Do not send passwords or recovery codes to anyone.
2. On GitHub create a **public** repository named `qwen3-8b-interpretability-guided-selective-lora`, initially without adding README, .gitignore, or LICENSE (they already exist here, except LICENSE intentionally).
3. Extract the ZIP and run the commands below from its extracted project directory. Replace `YOUR_GITHUB_USERNAME` with your actual username.

```bash
git init
git add .
git commit -m "Release reproducibility materials for Qwen3-8B selective LoRA study"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/qwen3-8b-interpretability-guided-selective-lora.git
git push -u origin main
```

GitHub may open a browser authentication flow or require a token/SSH configuration. Never share that token in chat. Alternatively connect GitHub to your AI assistant to assist with the upload after signing in and granting permissions.

Verify the public repo in a private/incognito browser before adding the permanent link to the manuscript. A tag/release plus archival DOI is advisable for a journal submission.
