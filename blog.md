---
layout: default
title: Blog
permalink: /blog/
---

# Blog

{% assign posts_by_year = site.posts | group_by_exp:"post", "post.date | date: '%Y'" %}
{% for year in posts_by_year %}
## {{ year.name }}
<div style="display: flex; flex-direction: column;">
  {% for post in year.items %}
  <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
    <span style="flex-grow: 1;"><a href="{{ post.url }}">{{ post.title }}</a></span>
    <span style="white-space: nowrap;">{{ post.date | date: "%b %d" }}</span>
  </div>
  {% endfor %}
</div>
{% endfor %}
