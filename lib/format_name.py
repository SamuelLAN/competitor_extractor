import re
import unicodedata
from urllib3.util import get_host

reg_remove_char = re.compile(r'[.,]')
reg_replace_char = re.compile(r'[\-`\'"\\]')
reg_space = re.compile(r'\s+')
reg_add_space_for_char = re.compile(r'([&])')
reg_name_suffix = re.compile(
    r'\s+(LLC|LTD|INC|CO|CL|LP|CORP|CORPORATION|C|SL|Corporation|Incorporated|PLLC|HLDG|HOLDINGS|HOLDING|HLDGs|HO|H|PLC|COMPANY|PTY|Global|GL|Canada|ENTERPRISES?|MANUFACTURING|LIMITED|MFG|INDUSTRIES|MACHINING|GROUP|GRP|USA|North America|G|GP)$',
    re.IGNORECASE)
reg_spots = re.compile(r'\.+')
reg_domain = re.compile(r'([^.]+)\.[^.]+$')
reg_not_en = re.compile(r'[^a-zA-Z0-9]', re.IGNORECASE)

reg_replace_space = re.compile(r'[\-_~\\]')
reg_comp = re.compile(r'[^a-zA-Z0-9& ]')

reg_http = re.compile('http(s)?://', re.IGNORECASE)
reg_path = re.compile(r'(/[a-zA-Z0-9_\-.]+)+', re.IGNORECASE)
reg_hash = re.compile(r'(#[a-zA-Z0-9_\-.]+)+', re.IGNORECASE)
reg_query = re.compile(r'\?[a-zA-Z0-9_\-.=&]+', re.IGNORECASE)
reg_sess = re.compile(r'(;[a-zA-Z0-9_\-.=&]+)+', re.IGNORECASE)


def full_2_half(_string):
    return unicodedata.normalize("NFKC", _string)


def company_name(name):
    name = reg_remove_char.sub('', name.lower().strip())
    name = reg_replace_char.sub(' ', name)
    name = name.replace(' and ', ' & ')
    name = reg_add_space_for_char.sub(r' \1 ', name)
    name = reg_space.sub(' ', name).strip()
    name = reg_name_suffix.sub('', reg_name_suffix.sub('', name))
    name = reg_space.sub(' ', name).strip()

    # name = reg_not_en.sub('', name)
    return name


# regularize the company name
def regular_comp_name(_name):
    _name = full_2_half(_name).strip().replace('"', "'").upper().replace(' AND ', ' & ')
    _name = reg_replace_space.sub(' ', _name)
    _name = reg_comp.sub('', _name)
    return reg_space.sub(' ', _name).strip()


def domain(url):
    url = reg_spots.sub('.', url)
    _domain = get_host(url)[1].strip('.').strip()
    ret_domain = reg_domain.findall(_domain)
    _domain = ret_domain[0] if ret_domain else ''
    return reg_not_en.sub('', _domain.lower())


def is_domain(_domain):
    if not _domain.strip() or '.' not in _domain or '/' in _domain or '?' in _domain or '#' in _domain or '=' in _domain \
            or ';' in _domain or _domain.split('.')[-1].lower() in ['php', 'html', 'asp', 'aspx', 'js', 'css']:
        return False
    return True


def get_domain(_url):
    try:
        return get_host(_url)[1]
    except:
        return reg_sess.sub('',
                            reg_query.sub('', reg_hash.sub('', reg_path.sub('', reg_http.sub('', _url.lower()).strip(
                                '/')))).strip('/'))


def get_first_domain(_url):
    _domain = get_domain(_url)
    return '.'.join(_domain.split('.')[-2:])
