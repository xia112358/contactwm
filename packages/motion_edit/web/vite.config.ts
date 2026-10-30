import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    outDir: '../motion_edit/web_dist',
    // Obsolete bundles are itemized and retired with gio trash, never erased by a build.
    emptyOutDir: false,
  },
});
