import os
import re
import itertools
import pickle

from tqdm import tqdm
from pythonopensubtitles.opensubtitles import OpenSubtitles


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class OST_LANG:
    en = 'eng'
    jp = 'jpn'
    all = 'all'


class OpenSubtitlesManager:
    USERNAME = os.getenv('OST_USERNAME')
    PASSWORD = os.getenv('OST_PASSWORD')

    def __init__(self, lang=OST_LANG.en, dist='.data'):
        self.lang = lang
        self.dist = os.path.join(BASE_DIR, dist)
        self.client = OpenSubtitles()

    def login(self):
        token = self.client.login(self.USERNAME, self.PASSWORD)
        assert type(token) == str
        return token

    def search(self, query='matrix'):
        data = self.client.search_subtitles([dict(sublanguageid=str(self.lang), query=query)])
        if data is None:
            return []
        subtitle_ids = [(d['IDSubtitleFile'], d['SubFormat']) for d in data]
        return subtitle_ids

    def ids_filter(self, subtitle_ids, chunk):
        def zipper(l):
            iteration = itertools.zip_longest(*[iter(l)]*chunk)
            return [[i for i in iter if i is not None] for iter in iteration]
        extensions = dict()
        for (sid, ext) in subtitle_ids:
            extensions[ext] = extensions.get(ext, []) + [sid]
        for ext, ids in extensions.items():
            extensions[ext] = zipper(ids)
        return extensions.items()

    def check_exists(self, subtitle_ids, ext):
        return [sid for sid in subtitle_ids
                    if not os.path.isfile(os.path.join(self.dist, '{}.{}'.format(sid, ext)))]

    def download_subtitles(self, chunk=5):
        if not os.path.exists(self.dist):
            os.makedirs(self.dist)
        token = self.login()
        subtitle_ids = self.search()
        iter = self.ids_filter(subtitle_ids, chunk)
        for ext, chunked in tqdm(iter):
            for ids in chunked:
                ids = self.check_exists(ids, ext)
                if len(ids) > 0:
                    self.client.download_subtitles(ids, output_directory=self.dist, extension=ext)

    def reglob(self, path, exp, invert=False):
        m = re.compile(exp)
        if invert is False:
            res = [f for f in os.listdir(path) if m.search(f)]
        else:
            res = [f for f in os.listdir(path) if not m.search(f)]
        res = [os.path.join(path, f) for f in res]
        return res

    def readfile(self, fpath):
        ext = fpath.split('.')[-1]
        with open(fpath, 'r') as f:
            content = f.readlines()
        return content, ext

    def should_ignore(self, line, patterns, invert=False):
        if line == '' or line == '&nbsp':
            return True
        for pattern in patterns:
            m = re.search(pattern, line)
            matched = m is not None
            if invert and not matched:
                return True
            if not invert and matched:
                return True
        return False

    def remove_strings(self, line, patterns):
        result = line
        for pattern in patterns:
            m = re.search(pattern, result)
            if m is not None:
                result = m.group(1)
        return result

    def parse_srt(self, content):
        ignores = [r'^[0-9:,]+ --> [0-9:,]+$', r'^[0-9]+$']
        removes = [r'\<[a-z]+\>(.*)\<\/[a-z]+\>']
        data = [self.remove_strings(line.replace('\n', ''), removes)
                for line in content
                if not self.should_ignore(line, ignores)]
        data = [line for line in data if line != '']
        return data

    def parse_sub(self, content):
        removes = [r'{[0-9]+}{[0-9]+}(.*)']
        data = [self.remove_strings(before_remove(line, ['\n', '</i>']), removes)
                for line in content]
        data = [line for line in data if line != '']
        return data

    '''ややこしいのでカット'''
    # def parse_ssa(self, content):
    #     ignores = [r'^Dialogue: (.*)']
    #     removes = [r'^Dialogue: (.*)']
    #     idx = content.index('[Events]')
    #     data = [self.remove_strings(line.replace('\n', ''), removes).split(',')
    #             for line in content[idx+3:]
    #             if not self.should_ignore(line, ignores, invert=True)]
    #     print(data)
    #     return data

    def before_remove(self, text, l):
        result = text
        for i in l:
            result = result.replace(i, '')
        return result

    def parse_smi(self, content):
        ignores = [r'\<SYNC Start\=[0-9]+\>\<P Class\=[A-Z]+\>(.*)']
        removes = [r'\<SYNC Start\=[0-9]+\>\<P Class\=[A-Z]+\>(.*)']
        data = [self.remove_strings(self.before_remove(line, ['\n', '<br>', '&nbsp']), removes)
                for line in content
                if not self.should_ignore(line, ignores, invert=True)]
        data = [line for line in data if line != '']
        return data

    def parse_txt(self, content):
        ignores = [r'^\[[A-Z ]+\]', r'[0-9:,]+']
        removes = ['[br]', '\n']
        data = [self.before_remove(line, removes)
                for line in content
                if not self.should_ignore(line, ignores)]
        data = [line for line in data if line != '']
        return data

    def get_sentenses(self):
        files = self.reglob(self.dist, r'\d+\.(srt|sub|smi|txt)$')
        data = []
        for file in files:
            # pickleファイルが存在する場合は、これをロードする
            ppath = '{}.pkl'.format(file)
            if os.path.exists(ppath):
                with open(ppath, 'rb') as f:
                    sentenses = pickle.load(f)
            else:
                content, ext = self.readfile(file)
                method_name = 'parse_{}'.format(ext)
                method = getattr(self, method_name)
                sentenses = method(content)
                with open(ppath, 'wb+') as f:
                    pickle.dump(sentenses, f)
            data.append(sentenses)
        return data

def open_subtitles(download=True, lang=OST_LANG.en, dist='.data'):
    manager = OpenSubtitlesManager(lang=lang, dist=dist)
    if download:
        print('[datasets] Cheking to download corpus datasets...')
        manager.download_subtitles()
    data = manager.get_sentenses()
    return data
