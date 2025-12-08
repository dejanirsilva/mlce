#!/bin/bash
# Copy files from _site to root for GitHub Pages deployment

# Copy index.html
cp _site/index.html .

# Copy site_libs directory
if [ -d "_site/site_libs" ]; then
  cp -r _site/site_libs .
fi

# Copy search.json if it exists
if [ -f "_site/search.json" ]; then
  cp _site/search.json .
fi

echo "Files copied from _site/ to root directory for GitHub Pages deployment"
