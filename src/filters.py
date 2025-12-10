import re


def pub_full_fmt(authors):

    def author_citation_format(author):
        affectations = re.search(r'\W+$', author)
        affectations = affectations.group(0) if affectations is not None else ''
        author = re.sub(r'\W+$', '', author)
        return f'{author}<sup>{affectations}</sup>'
    
    return ', '.join([author_citation_format(author) for author in authors])

def pub_author_fmt(authors, name='Nikhil Milind', max_authors=6):

    def author_citation_format(author):
        author_parts = author.strip().split()
        last_name = re.sub(r'\W+$', '', author_parts[-1])
        initials = ''.join([author_part[0] for author_part in author_parts[:-1]])
        affectations = re.search(r'\W+$', author)
        affectations = affectations.group(0) if affectations is not None else ''
        if re.sub(r'\W+$', '', author) == name:
            return f'<strong>{last_name} {initials}<sup>{affectations}</sup></strong>'
        return f'{last_name} {initials}<sup>{affectations}</sup>'

    if len(authors) <= max_authors:

        out_authors = list()

        for author in authors:

            out_authors.append(author_citation_format(author))

        return ', '.join(out_authors)

    else:

        out_authors_first = list()
        out_authors_last = list()

        first_three_authors = authors[:3]
        last_three_authors = authors[-3:]

        for author in first_three_authors:

            out_authors_first.append(author_citation_format(author))

        for author in last_three_authors:

            out_authors_last.append(author_citation_format(author))

        middle_part = '...'

        first_three_authors_stripped = [re.sub(r'\W+$', '', x) for x in first_three_authors]
        last_three_authors_stripped = [re.sub(r'\W+$', '', x) for x in last_three_authors]

        if name not in first_three_authors_stripped and name not in last_three_authors_stripped:
            if authors[3] == name:
                middle_part = f'{author_citation_format(name)}, ...'
            elif authors[-4] == name:
                middle_part = f'..., {author_citation_format(name)}'
            else:
                middle_part = f'..., {author_citation_format(name)}, ...'

        return ', '.join(out_authors_first) + f', {middle_part}, ' + ', '.join(out_authors_last)

    
