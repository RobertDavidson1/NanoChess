# NanoChess board

The chessboard is a static page. Its chess rules, board component, and piece
images load in the browser from external HTTPS URLs. No build step is required.
The Python model and training code do not run on GitHub Pages.

## GitHub Pages

1. Push `web/` to the `master` branch of your GitHub repository.
2. In the repository, open **Settings → Pages** and set **Source** to
   **Deploy from a branch**.
3. Select **master** and **/ (root)**, then click **Save**.

The board will be at `https://<owner>.github.io/<repository>/web/`.
Later pushes to `master` update the site automatically. If your branch is named
`main`, use that branch instead.

See [GitHub's Pages publishing instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).
