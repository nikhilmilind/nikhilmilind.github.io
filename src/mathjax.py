import markdown
import html


class MathJaxPattern(markdown.inlinepatterns.Pattern):

    def handleMatch(self, m):
        text = html.escape(m.group(2) + m.group(3) + m.group(2))
        return self.md.htmlStash.store(text)


class MathJaxExtension(markdown.extensions.Extension):
    def extendMarkdown(self, md):
        md.inlinePatterns.register(MathJaxPattern(r'(?<!\\)(\$\$?)(.+?)\2', md), 'mathjax', 185)


def makeExtension():
    return MathJaxExtension()
