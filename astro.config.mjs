// @ts-check
import { defineConfig } from 'astro/config';
import remarkToc from 'remark-toc'
import remarkMath from 'remark-math';
import rehypeMathjax from 'rehype-mathjax';

export default defineConfig({
  markdown: {
    remarkPlugins: [remarkMath, remarkToc],
    rehypePlugins: [rehypeMathjax],
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark',
      },
    },
  },
});

