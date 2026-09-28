import unittest
from html.parser import HTMLParser

class ReportExportTests(unittest.TestCase):
    def test_isolated_scores_have_visible_markers_without_connecting_gaps(self):
        from soci_ai.desktop.report_export import render_html
        page = render_html({'elapsed_seconds':3, 'history':[
            {'at_ms':0,'sbi':20}, {'at_ms':1000,'sbi':None}, {'at_ms':2000,'sbi':80}]})
        class Markers(HTMLParser):
            def __init__(self):
                super().__init__(); self.points=[]
            def handle_starttag(self, tag, attrs):
                values = dict(attrs)
                if tag=='circle' and values.get('class')=='isolated-sample':
                    self.points.append(values)
        parser = Markers(); parser.feed(page)
        self.assertEqual(len(parser.points), 2)
        self.assertTrue(all(point.get('fill')=='url(#risk)' for point in parser.points))

    def test_json_filter_changes_default_html_extension_and_content(self):
        from soci_ai.desktop import report_export
        try:
            path, content = report_export.export_payload({'average_sbi':30}, 'report.html', '结构化数据 (*.json)')
        except AttributeError:
            self.fail('export format must normalize the default extension')
        import json
        self.assertEqual(path, 'report.json')
        self.assertEqual(json.loads(content)['average_sbi'], 30)

    def test_readable_html_contains_summary_and_gradient_curve_without_running_scripts(self):
        try:
            from soci_ai.desktop import report_export
        except ImportError:
            self.fail('readable HTML export is not implemented')
        page = report_export.render_html({
            'elapsed_seconds':30, 'average_sbi':30, 'peak_sbi':80,
            'history':[{'at_ms':0,'sbi':5,'cues':[]},
                       {'at_ms':10000,'sbi':80,'cues':['疑似学历偏见']},
                       {'at_ms':20000,'sbi':None}, {'at_ms':30000,'sbi':10}],
            'advice':[{'title':'本场总结', 'body':'需要留意学历概括。'}],
            'transcripts':[{'at_ms':10000,'text':'<script>alert(1)</script>'}]})
        self.assertIn('需要留意学历概括', page)
        self.assertIn('<svg', page)
        self.assertIn('linearGradient', page)
        self.assertIn('疑似学历偏见', page)
        self.assertIn('&lt;script&gt;', page)
        self.assertNotIn('<script>', page)
        class Paths(HTMLParser):
            def __init__(self):
                super().__init__(); self.values=[]
            def handle_starttag(self, tag, attrs):
                if tag=='path':self.values.append(dict(attrs).get('d',''))
        parser=Paths();parser.feed(page)
        self.assertTrue(any(path.count('M')==2 for path in parser.values), 'missing score must break the line')
