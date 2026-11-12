"""Isolated workflow checks. Real vectors use a pre-warmed cache; Ollama transport is mocked."""
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import text
from document_search import create_app, db
from document_search.models import Document, Passage
from document_search import services


class Workflows(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.app = create_app({
            'TESTING': True,
            'SQLALCHEMY_DATABASE_URI': f'sqlite:///{self.root}/test.db',
            'UPLOAD_FOLDER': str(self.root / 'uploads'),
            'MODEL_CACHE_DIR': os.environ.get('EMBEDDING_CACHE_DIR', str(self.root / 'cache')),
            'EMBEDDING_OFFLINE': True,
            'OLLAMA_MODEL': 'classroom-local',
        })
        self.client = self.app.test_client()
        self.tags = patch('document_search.blueprints.chat.get_installed_models', return_value=([{'name': 'classroom-local'}], None))
        self.tags.start()
        self.search_tags = patch('document_search.blueprints.search.get_installed_models', return_value=([], 'offline test'))
        self.search_tags.start()

    def tearDown(self):
        self.tags.stop()
        self.search_tags.stop()
        with self.app.app_context():
            db.session.remove()
            db.engine.dispose()
        self.directory.cleanup()

    def upload(self, name, content):
        response = self.client.post('/documents/upload', data={'file': (io.BytesIO(content), name)})
        self.assertEqual(response.status_code, 302)
        location = response.location
        response.close()
        return location

    def test_formats_preview_and_errors(self):
        from docx import Document as Word
        from openpyxl import Workbook
        from pptx import Presentation
        from pptx.util import Inches
        from pypdf import PdfWriter
        samples = {'a.txt': b'known sentence\nsecond line', 'a.markdown': b'# Heading\n- item\n\n|A|B|\n|1|2|',
                   'a.htm': b'<h1>Visible</h1><p>html text</p><script>hidden_script</script>'}
        w = Word(); w.add_paragraph('Word paragraph'); w.add_table(rows=1, cols=2).cell(0, 0).text = 'Word table'
        data = io.BytesIO(); w.save(data); samples['a.docx'] = data.getvalue()
        w = Workbook(); w.active['A1'] = 'Hours'; w.active['B1'] = '=1+2'; w.create_sheet('Second')['C2'] = 'Second sheet'
        data = io.BytesIO(); w.save(data); samples['a.xlsx'] = data.getvalue()
        p = Presentation()
        for label in ['First slide', 'Second slide']:
            slide = p.slides.add_slide(p.slide_layouts[6]); slide.shapes.add_textbox(Inches(1), Inches(1), Inches(3), Inches(1)).text = label
        slide.shapes.add_table(1, 2, Inches(1), Inches(3), Inches(4), Inches(1)).table.cell(0, 0).text = 'Slide table'
        data = io.BytesIO(); p.save(data); samples['a.pptx'] = data.getvalue()
        pdf = PdfWriter(); pdf.add_blank_page(width=200, height=200); data = io.BytesIO(); pdf.write(data); samples['a.pdf'] = data.getvalue()
        for name, data in samples.items():
            location = self.upload(name, data)
            page = self.client.get(location)
            self.assertEqual(page.status_code, 200, name)
            with self.app.app_context():
                doc = Document.query.filter_by(original_name=name).one()
                self.assertIsNotNone(doc.extracted_sections)
                if name == 'a.xlsx':
                    self.assertIn('cached result: unavailable', doc.extracted_text)
                    self.assertIn('Second, row 2', doc.extracted_text)
                if name == 'a.pptx':
                    self.assertIn('slide 2', doc.extracted_text)
                    self.assertIn('Slide table', doc.extracted_text)
                if name == 'a.htm':
                    self.assertNotIn('hidden_script', doc.extracted_text)
                if name == 'a.pdf':
                    self.assertEqual(doc.status, 'empty')
        self.upload('empty.txt', b'')
        with self.app.app_context():
            self.assertEqual(Document.query.filter_by(original_name='empty.txt').one().status, 'empty')
            before = Document.query.count()
        for name, data in [('bad.jpg', b'bad'), ('bad.docx', b'invalid zip'), ('bad.xls', b'invalid excel')]:
            self.assertEqual(self.upload(name, data), '/documents')
        self.upload('duplicate.txt', samples['a.txt'])
        with self.app.app_context():
            self.assertEqual(Document.query.count(), before)
        self.assertEqual(self.client.get('/documents/1/delete').status_code, 405)
        self.assertEqual(self.client.post('/documents/upload', data={'file': (io.BytesIO(b'x'*(10*1024*1024+1)), 'large.txt')}).status_code, 302)

    @unittest.skipUnless(os.environ.get('EMBEDDING_CACHE_DIR'), 'Set EMBEDDING_CACHE_DIR to a warm cache for real-vector test')
    def test_real_vectors_approved_text_persistence_delete(self):
        original = ('The laboratory password is moss lighthouse. La password del laboratorio e moss lighthouse.\n' * 100) + 'TAIL_ZEBRA'
        location = self.upload('lesson.txt', original.encode())
        with self.app.app_context():
            doc = Document.query.one(); doc_id = doc.id
            Path(self.app.config['UPLOAD_FOLDER'], doc.storage_name).write_text('CHANGED AFTER PREVIEW')
        self.client.post(f'/documents/{doc_id}/index')
        with self.app.app_context():
            passages = Passage.query.all()
            self.assertGreater(len(passages), 5)
            self.assertTrue(any('TAIL_ZEBRA' in p.text for p in passages))
            self.assertTrue(all('CHANGED AFTER PREVIEW' not in p.text for p in passages))
            self.assertTrue(all(len(json.loads(p.embedding)) == 384 for p in passages))
            self.assertTrue(all(len(services.get_tokenizer().encode(p.text).ids) <= 128 for p in passages))
            self.assertTrue(services.keyword_search('lighthouse'))
            self.assertEqual(services.keyword_search('!!!???'), [])
            self.assertEqual(services.keyword_search('" OR - :'), [])
            results, error = services.semantic_search('Qual e la password del laboratorio?')
            self.assertIsNone(error); self.assertIn('password', results[0]['text'])
            print('REAL VECTOR RESULT:', results[0]['similarity'], results[0]['page_or_section'])
            self.assertEqual(db.session.execute(text('PRAGMA foreign_keys')).scalar(), 1)
            db.session.remove(); db.engine.dispose()
            self.assertIn('TAIL_ZEBRA', db.session.get(Document, doc_id).extracted_text)
            self.assertTrue(services.keyword_search('lighthouse'))
        self.assertIn(b'TAIL_ZEBRA', self.client.get(location).data)
        self.client.post(f'/documents/{doc_id}/delete')
        with self.app.app_context():
            self.assertEqual(Passage.query.count(), 0)
            self.assertEqual(services.keyword_search('lighthouse'), [])

    def test_chat_modes_isolation_and_settings(self):
        with patch('document_search.blueprints.chat.call_chat_completion', return_value=('Transport test answer', None)) as call, patch('document_search.blueprints.chat.semantic_search', side_effect=AssertionError('General chat must not retrieve')):
            self.assertIn(b'General chat', self.client.get('/chat').data)
            self.client.post('/chat', data={'message': 'Hello', 'use_documents': '0'})
            self.client.post('/chat', data={'message': 'Follow-up', 'use_documents': '0'})
            messages = call.call_args.kwargs['messages']
            self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'user'])
            self.assertNotIn('Source Passages', str(messages))
            other = self.app.test_client(); self.assertNotIn(b'Transport test answer', other.get('/chat').data)
            self.client.get('/chat?use_documents=1')
            self.assertNotIn(b'Transport test answer', self.client.get('/chat?use_documents=0').data)
            self.client.post('/chat', data={'message': 'Again', 'use_documents': '0'})
            self.client.post('/chat', data={'new_chat': '1', 'use_documents': '0'})
            self.assertNotIn(b'Transport test answer', self.client.get('/chat').data)
        with patch('document_search.blueprints.chat.semantic_search', return_value=(None, 'model unavailable')), patch('document_search.blueprints.chat.keyword_search', side_effect=AssertionError('No silent fallback')):
            self.client.post('/chat', data={'message': 'Evidence?', 'use_documents': '1'})
            self.assertIn(b'Choose keyword retrieval explicitly', self.client.get('/chat?use_documents=1').data)
        self.assertEqual(self.client.get('/chat/settings').status_code, 200)
        invalid = self.client.post('/chat/settings', data={'server': 'file:///tmp', 'action': 'check'})
        self.assertIn(b'Enter a server URL', invalid.data)
        checked = self.client.post('/chat/settings', data={'server': 'http://192.168.1.5:11434', 'action': 'check'})
        self.assertIn(b'classroom-local', checked.data)
        with patch('dotenv.set_key') as save:
            response = self.client.post('/chat/settings', data={'server': 'http://192.168.1.5:11434', 'model': 'classroom-local', 'action': 'save'})
            self.assertEqual(response.status_code, 302); self.assertEqual(save.call_count, 2)
            self.assertEqual(self.app.config['OLLAMA_BASE_URL'], 'http://192.168.1.5:11434')

    def test_keyword_only_when_embeddings_fail(self):
        self.upload('plain.txt', b'Unique keyword search survives embedding failure.')
        with patch('document_search.services._get_embedding_model', return_value=None):
            self.client.post('/documents/1/index')
            with self.app.app_context():
                self.assertTrue(services.keyword_search('Unique'))
                self.assertIsNone(Passage.query.first().embedding)
            response = self.client.post('/search', data={'q': 'Unique', 'retrieval_method': 'semantic'})
            self.assertIn(b'Choose keyword retrieval explicitly', response.data)


if __name__ == '__main__':
    unittest.main(verbosity=2)
