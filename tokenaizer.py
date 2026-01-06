import mwparserfromhell as mwp
import bz2
from collections import Counter
from xml.etree import ElementTree
from nltk.corpus import stopwords


import re
import gdown


def page_iter(wiki_file):
  """ Reads a wiki dump file and create a generator that yields pages.
  Parameters:
  -----------
  wiki_file: str
    A path to wiki dump file.
  Returns:
  --------
  tuple
    containing three elements: article id, title, and body.
  """
  # open compressed bz2 dump file
  with bz2.open(wiki_file, 'rt', encoding='utf-8', errors='ignore') as f_in:
    # Create iterator for xml that yields output when tag closes
    elems = (elem for _, elem in ElementTree.iterparse(f_in, events=("end",)))
    # Consume the first element and extract the xml namespace from it.
    # Although the raw xml has the  short tag names without namespace, i.e. it
    # has <page> tags and not <http://wwww.mediawiki.org/xml/export...:page>
    # tags, the parser reads it *with* the namespace. Therefore, it needs the
    # namespace when looking for child elements in the find function as below.
    elem = next(elems)
    m = re.match(r"^{(http://www\.mediawiki\.org/xml/export-.*?)}", elem.tag)
    if m is None:
        raise ValueError("Malformed MediaWiki dump")
    ns = {"ns": m.group(1)}
    page_tag = ElementTree.QName(ns['ns'], 'page').text
    # iterate over elements
    for elem in elems:
      if elem.tag == page_tag:
        # Filter out redirect and non-article pages
        if elem.find('./ns:redirect', ns) is not None or \
           elem.find('./ns:ns', ns).text != '0':
          elem.clear()
          continue
        # Extract the article wiki id
        wiki_id = elem.find('./ns:id', ns).text
        # Extract the article title into a variables called title
        title_elem = elem.find('./ns:title', ns) # extract title element from ns
        title = title_elem.text if title_elem is not None else ""
        # extract body
        body = elem.find('./ns:revision/ns:text', ns).text

        yield wiki_id, title, body
        elem.clear()

def filter_article_links(title):
  """ Return false for wikilink titles (str) pointing to non-articles such as images, files, media and more (as described in the documentation).
      Otherwise, returns true.
  """
  # Filter out empty or whitespace-only titles or internal section links
  if not title or not title.strip() or title.startswith('#'):
    return False

  # If there's no colon, it's a regular article
  if ':' not in title:
    return True

  # non_article prefixes list
  non_article_prefixes = [
        'file', 'image', 'category', 'wikipedia',
        'help', 'template', 'user', 'talk',
        'media', 'special'
    ]

  pars = title.split(":")[0].lower()
  return pars not in non_article_prefixes


def get_wikilinks(wikicode):
  """ Traverses the parse tree for internal links and filter out non-article
  links.
  Parameters:
  -----------
  wikicode: mwp.wikicode.Wikicode
    Parse tree of some WikiMedia markdown.
  Returns:
  --------
  list of (link: str, anchor_text: str) pair
    A list of outgoing links from the markdown to wikipedia articles.
  """
  links = []
  for wl in wikicode.ifilter_wikilinks():
    # skip links that don't pass our filter
    title = str(wl.title)
    if not filter_article_links(title):
      continue
    # if text is None use title, otherwise strip markdown from the anchor text.
    text = wl.text
    if text is None:
      text = title
    else:
      text = text.strip_code()
    # remove any lingering section/anchor reference in the link
    # YOUR CODE HERE
    title = title.split('#')[0] # Remove anchors from title
    text = str(text).split('#')[0] # Remove anchors from text
    links.append((title, text))
  return links



def remove_markdown(text):
  return mwp.parse(text).strip_code()


def get_html_pattern():
  # Pattern breakdown:
    # <       - opening angle bracket
    # [^>]+   - one or more characters that are NOT >
    # >       - closing angle bracket
  return r'<[^>]+>'


def get_date_pattern():
    # Days 1-31
    day_31 = r'(0?[1-9]|[12][0-9]|3[01])'

    # Days 1-30
    day_30 = r'(0?[1-9]|[12][0-9]|30)'

    # Days 1-29 (for February)
    day_29 = r'(0?[1-9]|[12][0-9])'

    # Months with 31 days
    months_31 = r'(Jan|January|Mar|March|May|Jul|July|Aug|August|Oct|October|Dec|December)'

    # Months with 30 days
    months_30 = r'(Apr|April|Jun|June|Sep|September|Nov|November)'

    # February
    feb = r'(Feb|February)'

    # MM DD YYYY
    text_31 = rf'{months_31}\s+{day_31}(?:st|nd|rd|th)?,?\s*\d{{2,4}}'
    text_30 = rf'{months_30}\s+{day_30}(?:st|nd|rd|th)?,?\s*\d{{2,4}}'
    text_feb = rf'{feb}\s+{day_29}(?:st|nd|rd|th)?,?\s*\d{{2,4}}'

    # DD MM YYYY
    text2_31 = rf'{day_31}(?:st|nd|rd|th)?\s+{months_31},?\s+\d{{2,4}}'
    text2_30 = rf'{day_30}(?:st|nd|rd|th)?\s+{months_30},?\s+\d{{2,4}}'
    text2_feb = rf'{day_29}(?:st|nd|rd|th)?\s+{feb},?\s+\d{{2,4}}'

    # Numeric dates
    numeric_date = rf'{day_29}[/.\-]{day_29}[/.\-]\d{{2,4}}'

    # combine all patterns
    return rf'{text_31}|{text_30}|{text_feb}|{text2_31}|{text2_30}|{text2_feb}|{numeric_date}'

def get_time_pattern():
  # valid hours
    hours_24 = r'([01]?[0-9]|2[0-3])'  # 0-23
    hours_12 = r'(0?[1-9]|1[0-2])'      # 1-12

  # valid minutes and seconds
    min_sec = r'[0-5][0-9]' # 00-59

  # 24-hour without AM/PM
    time_24 = rf'(?<!\d:){hours_24}:{min_sec}(:{min_sec})?(?!:\d{{2}}|\s?[aApP]\.?[mM]\.?)'

  # 12-hour DOT + AM/PM
    time_12_dot = rf'{hours_12}\.{min_sec}[AP]M'

  # 12-hour NO SEP + a.m./p.m.
    time_12_nosep = rf'{hours_12}{min_sec}[ap]\.m\.'

   # combine all patterns
    return rf'{time_24}|{time_12_dot}|{time_12_nosep}'

def get_percent_pattern():
  # Pattern breakdown:
    # \d+  - one or more digits (without)
    # \.?  - optional decimal point
    # \d*  - zero or more digits after decimal
    # %    - literal percent sign
  return r'\d+\.?\d*%'


def get_number_pattern():
  # Pattern breakdown:
    # (?<![A-Za-z0-9_+\-,.])   - number cannot be preceded by letters, digits, signs, comma, or dot
    # (?:[+-])?                - optional sign
    # ([0-9]{1,3}(,[0-9]{3})*|[0-9]+) - integer part: valid commas or plain digits
    # (?:\.[0-9]+)?            - optional decimal part
    # (?!%|\.\d|\.[A-Za-z]|,\d|\d)-no percent, digit after dot/comma, or letters""
    sign = r"(?:[+-])?"
    int_part = r"([0-9]{1,3}(,[0-9]{3})*|[0-9]+)(?:\.[0-9]+)?"
    prefix = r"(?<![A-Za-z0-9_\+\-\,\.])"
    suffix = r"(?!%)(?!\.\d)(?!\.[A-Za-z])(?!,\d)(?!\d)"

    # combine all patterns
    return rf"{prefix}{sign}{int_part}{suffix}"


def get_word_pattern():
  # Pattern breakdown:
    # (?<!-)           - not preceded by hyphen
    # [a-zA-Z]+        - one or more letters (start)
    # (['-][a-zA-Z]+)* - zero or more groups of (apostrophe/hyphen + letters)
    # '?               - optional trailing apostrophe (for parents')
  return r"(?<!-)[a-zA-Z]+(['-][a-zA-Z]+)*'?"


RE_TOKENIZE = re.compile(rf"""
(
    # parsing html tags
     (?P<HTMLTAG>{get_html_pattern()})
    # dates
    |(?P<DATE>{get_date_pattern()})
    # time
    |(?P<TIME>{get_time_pattern()})
    # Percents
    |(?P<PERCENT>{get_percent_pattern()})
    # Numbers
    |(?P<NUMBER>{get_number_pattern()})
    # Words
    |(?P<WORD>{get_word_pattern()})
    # space
    |(?P<SPACE>[\s\t\n]+)
    # everything else
    |(?P<OTHER>.))""",  re.MULTILINE | re.IGNORECASE | re.VERBOSE | re.UNICODE)

# A counter mapping article id to number of page views
wid2pv = Counter({
    '17324616': 10, '17324662': 4, '17324672': 16, '17324677': 612,
    '17324689': 66, '17324702': 274, '17324704': 49, '17324943': 76,
    '17324721': 35, '17324736': 2801, '17324747': 641, '17324758': 33,
    '17324768': 26, '17324783': 28, '17324788': 575, '17324790': 43,
    '17324802': 29, '17324816': 159, '17324818': 57, '17324823': 60,
    '17324834': 19, '17324835': 7, '17324893': 116, '17324908': 2038,
    '15580374': 181126232, '1610886': 4657885, '30635': 8143874,
    '3390': 4525604, '49632909': 5027640, '51150040': 3284643,
    '60827': 4323859, '623737': 3427102, '65984422': 3733064, '737': 6039676
})


def most_viewed(pages):
  """Rank pages from most viewed to least viewed using the above `wid2pv`
     counter.
  Parameters:
  -----------
    pages: An iterable list of pages as returned from `page_iter` where each
           item is an article with (id, title, body)
  Returns:
  --------
  A list of tuples
    Sorted list of articles from most viewed to least viewed article with
    article title and page views. For example:
    [('Langnes, Troms': 16), ('Langenes': 10), ('Langenes, Finnmark': 4), ...]
  """
  # Empty list for all tuples
  title_views_list = []
  for page_id, title, body in pages:
    page_views = wid2pv.get(page_id, 0) # get page info
    title_views_list.append((title, page_views)) # add tuple to list
    sorted_by_views = sorted(title_views_list, key=lambda item: item[1], reverse=True) # sort by page views in descending order
  return sorted_by_views

def tokenize(text):
  return [(v, k) for match in RE_TOKENIZE.finditer(text)
                 for k, v in match.groupdict().items()
                 if v is not None and k != 'SPACE']