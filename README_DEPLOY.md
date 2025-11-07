# Deploying UCL 23-24 Explorer to Streamlit Cloud

## Repo structure
Place these files at the root of your GitHub repo:
- `app_final.py`  (the Streamlit app)
- `requirements.txt`
- `runtime.txt` (optional, forces Python version)
- `.streamlit/config.toml` (optional UI/server settings)
- `README_DEPLOY.md` (this file)

## Steps to deploy
1. Create a **new GitHub repository** and push these files to the repository's `main` branch:
   ```bash
   git init
   git add app_final.py requirements.txt runtime.txt .streamlit/config.toml README_DEPLOY.md
   git commit -m "Add UCL explorer app for Streamlit Cloud"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

2. Go to **Streamlit Cloud** (https://streamlit.io/cloud) and click **'New app'** → choose the GitHub repo and branch (`main`) and set the entrypoint to `app_final.py`.

3. Click **Deploy**. Streamlit Cloud will install packages from `requirements.txt` and launch the app.

## Notes & troubleshooting
- If you see import errors, ensure `requirements.txt` includes all dependencies and the file is in the repo root.
- If `requests` calls to FBref or UEFA fail on Streamlit Cloud due to remote blocking or rate limits, use the Upload CSV page to provide pre-downloaded CSVs instead.
- Large files (CSV) should be stored in Git LFS or a cloud storage (S3, Google Drive) and downloaded at runtime if needed.
- For persistent storage of generated CSVs, consider using an external storage service — Streamlit Cloud's filesystem is ephemeral across deploys.

## Optional: add secrets
If you later add APIs that require keys, go to the Streamlit Cloud app dashboard → Settings → Secrets and add them there. Access them in your app with `st.secrets`.
