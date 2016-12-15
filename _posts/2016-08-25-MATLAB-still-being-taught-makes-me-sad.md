---
layout: post
title: Universities Have to Emancipate From Using MATLAB NOW!
---

**DISCLAIMER:** This is written from my perspective as an engineering student (!), still having a lot to learn.

I think teaching MATLAB to engineering students is one of the most harmful mistakes universities make in the education of their students.
There's a lot of interesting and clever critique about MATLAB as a language from people, who really know what they are talking about.
For example:
<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">You need to stop teaching your students Matlab now <a href="http://t.co/9t2DMA1yJI">http://t.co/9t2DMA1yJI</a></p>&mdash; Régis Behmo (@regisb) <a href="https://twitter.com/regisb/status/506422540236251136">September 1, 2014</a></blockquote>
<script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>

But to me the most important point why I consider MATLAB taught to students very harmful is something different:

**MATLAB is almost exclusively used by engineers.**

That's why you never get out your bubble of other engineers and get the chance to learn from other professions like programmers, computer scientists about crafting code. MATLAB makes it very easy to use it as an super advanced calculator or to write spaghetti-code scripts.
I think there is nothing bad about it and I can imagine who great MATLAB was, connecting an interpreted script-language with the access to FORTRAN libraries for extremely fast matrix computations. I think being able to play with data in an exploratory manner, trying things out on the fly, without makefiles, compiling stuff all the time etc. is very important in some scientific research.

But because MATLAB is completely isolated from everything outside engineering, you never even get the chance to realize what you don't know:

* you won't get exposed to version control
* you don't learn about the benefits of open source
* you don't get advice from people who don't own some really expensive MATLAB license
* there's not much public code from people, who actually are crafting code for a living you can learn from to get better
* you don't learn about good documentation practices, writing unit tests, SOLID-principles etc. etc.

Of course everything above is also possible with MATLAB. But I had to learn the hard way that you don't even know what's out there.
It is perfectly fine if someone just needs some calculator on steroids but as soon as there's a need to scale, to collaborate, to involve other people, to make the code reliable, you suddenly hit a wall. And most of the times, you don't know that you will get there as you just start a project.

If there wouldn't be alternatives like Python with Numpy/Scipy maybe the advantages of MATLAB would be worth the sacrifices. But these times I think it is really hurtful that universities still teach MATLAB to students, because this enforces separation instead of collaboration and if the universities can't support the spread of free software, making it easier to communicate with other fields of science, who will?
