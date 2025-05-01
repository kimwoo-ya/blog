# My Blog Repo

## Installations.
```bash
# install hugo
$ brew install hugo

# checks hugo versions...
$ hugo version
hugo v0.147.0+extended+withdeploy darwin/arm64 BuildDate=2025-04-25T15:26:28Z VendorInfo=brew
```

## Quick start
### Procject Initialization...
```bash
# make git repo
git remote add origin https://github.com/kimwoo-ya/wooya-github-io.git

# make hugo-project
$ hugo new site my_blog
# Initialize git repository
$ git init
# add hugo theme
$ git submodule add https://github.com/janraasch/hugo-bearblog.git themes/hugo-bearblog
# insert theme option
$ echo "theme = 'hugo-bearblog'" >> hugo.toml
```
### Make new posts..
```bash
# make posts....
$ hugo new posts/my-first-post.md
$ hugo new posts/my-second-post.md
$ hugo new "code-test/백준/2798.md"

# start hugo-server(local)
$ hugo server -D

# start hugo-server
# modify post.md(draft) : true -> false
$ hugo server
```
## Changing Theme..
```bash
$ rm -rf themes/hugo-bearblog


$ git submodule deinit themes/hugo-bearblog
$ git rm themes/hugo-bearblog
$ rm .gitmodules

$ touch .gitmodules; git submodule add https://github.com/joeroe/risotto themes/risotto

# change defined theme options...
$ vim hugo.toml

# add comment feature
$ mkdir -p layouts/partials
$ vim utterances.html
$ mkdir -p layouts/_default
$ touch layouts/_default/single.html
```


## Destinations..
[kimwoo-ya.github.io/blog/](https://kimwoo-ya.github.io/blog)