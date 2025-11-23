
def pub_author_fmt(authors, name='Nikhil Milind', max_authors=6):

    def author_citation_format(author):
        author_parts = author.strip().split()
        last_name = author_parts[-1]
        initials = ''.join([author_part[0] for author_part in author_parts[:-1]])
        if author == name:
            return f'<strong>{last_name} {initials}</strong>'
        return f'{last_name} {initials}'

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

        if name not in first_three_authors and name not in last_three_authors:
            if authors[3] == name:
                middle_part = f'{author_citation_format(name)}, ...'
            elif authors[-4] == name:
                middle_part = f'..., {author_citation_format(name)}'
            else:
                middle_part = f'..., {author_citation_format(name)}, ...'

        return ', '.join(out_authors_first) + f', {middle_part}, ' + ', '.join(out_authors_last)

    
