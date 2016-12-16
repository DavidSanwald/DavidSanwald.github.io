---
layout: post
title: Testing and Deploying With Travis CI and Some Other Blog Stuff
subtitle: Also Giving Credit and Trying to be Self-aware
---
My first threesome experience with Jekyll and Travis CI, it's been a rough start but now we're getting along. And this blog has some new quirky colors. In case you wonder where they're from:
<div class="tumblr-post" data-href="https://embed.tumblr.com/embed/post/_YHa9p7lUt4cZBPOOUOuVQ/110716093015" data-did="21b67e78d15d149f0d55811972d4a27a32d346d1"><a href="http://wesandersonpalettes.tumblr.com/post/110716093015/ash-should-we-dance">http://wesandersonpalettes.tumblr.com/post/110716093015/ash-should-we-dance</a></div>  <script async src="https://assets.tumblr.com/post.js"></script>
<!--more-->

## Tool Addiction and Life Lessons
I think I'm not the only one being sort of addicted to trying new shiny things, automating anything or having to use something just because it seems cool.
Of course there are many things right about continuous integration but the benefits for my tiny blog with it's very moderate post frequency? Maybe mainly educational (:
I'm using the [Jekyll-Scholar](https://github.com/inukshuk/jekyll-scholar) plugin, which isn't one of the very few plugins supported by GitHub. So I had to build this blog locally. I used a small Rakefile to compile everything on a source branch and pushing the compiled site to GitHub. It worked perfectly fine. The only downside was, side I couldn't just write some markdown file from every computer or handy and let GitHub build a blog post from it. But this is more hypothetically in my own use case (note to myself: should post more often).
That it worked just fine didn't keep me from switching to remotely building the blog using Travis CI.
As long as everything works fine, it's relatively straight forward. This [article](https://www.linux.com/learn/how-automate-web-application-testing-docker-and-travis) was really helpful, although it also employs Docker and uses a Flask App example.

However if things don't go so smooth...
In the end it took my a few hours till the first working build on Travis. Somehow my source branch didn't find the master branch in its refspec, no matter what I tried to update the references. Furthermore my Rakefile refused to find the default environment variables from Travis CI as well as the variables read from my encrypted access token in the .travis.yml file.

Whatever I'm to new to this to share some clever details about getting your Jekyll builds to pass. But there are two tings:

1. After a few hours fighting with Travis CI and Git I was so hacked off that I accidentally ended up leaving a decrypted access token in some code I copy pasted in a cry for help on stackoverflow. I realized it two minutes afters posting, quickly revoked the token on GitHub. Still not one of my proudest moments but a really humbling life lesson.
2. I want to give credit, to [Martin Fenner](https://github.com/mfenner) because I'm using his Rakefile, which I modified to include testing with [html-proofer](https://github.com/gjtorikian/html-proofer). I also want share [this](http://jasonseifer.com/2010/04/06/rake-tutorial) extremely clear Rake tutorial, which really was of great help to me.

## Feeling Foxy

I've been an admirer of the [Wes Anderson Palettes](http://wesandersonpalettes.tumblr.com/) tumblr for a long time. [Palettable](https://jiffyclub.github.io/palettable/) made some of them available for scientific plotting with Python and this is also where I discovered the tumblr. I also really like foxes. So I just HAD to use this palette:
<div class="tumblr-post" data-href="https://embed.tumblr.com/embed/post/_YHa9p7lUt4cZBPOOUOuVQ/110716093015" data-did="21b67e78d15d149f0d55811972d4a27a32d346d1"><a href="http://wesandersonpalettes.tumblr.com/post/110716093015/ash-should-we-dance">http://wesandersonpalettes.tumblr.com/post/110716093015/ash-should-we-dance</a></div>  <script async src="https://assets.tumblr.com/post.js"></script>
Besides the colors I'm still enjoying the [type](https://github.com/rohanchandra/type-theme) theme. Just changed a few minor things like using Fire Sans as first font if available and switching from KaTeX to MathJax because I was missing a lot of important font options even though KaTeX has much faster rendering.

## Musical Crush for the Weekend

Last week I was as the PWR BTTM concert. They played with Spook School a queer band from Scotland and both were absolutely amazing. I not only love the music but also think bands like this are important, especially in the light of recent political developments. So here's their Tiny Desk Concert, which I love so much:

<iframe width="1120" height="630" src="https://www.youtube.com/embed/ji-EdRtL9qU" frameborder="0" allowfullscreen></iframe>
