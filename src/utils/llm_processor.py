import logging
import re
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import json

logger = logging.getLogger(__name__)

class LLMContentProcessor:
    """Advanced content processing specifically optimized for LLM consumption"""
    
    def __init__(self):
        self.max_summary_length = 500
        self.max_key_points = 8
        self.min_content_length = 100
        
        # Code detection patterns
        self.code_patterns = [
            (r'```(\w+)?\n(.*?)\n```', 'code_block'),
            (r'<code>(.*?)</code>', 'inline_code'),
            (r'<pre>(.*?)</pre>', 'preformatted'),
            (r'^\s{4,}.*', 'indented_code')
        ]
        
        # Key information extraction patterns
        self.fact_patterns = [
            r'According to .+?,',
            r'Research shows .+?\.',
            r'Studies indicate .+?\.',
            r'The key finding .+?\.',
            r'Important: .+?\.',
            r'Note: .+?\.',
        ]
        
        # Citation patterns
        self.citation_patterns = [
            r'\(.*?\d{4}.*?\)',  # (Author, 2024)
            r'\[.*?\]',          # [1] or [Source]
            r'Source: .+',       # Source: ...
            r'According to .+?(?=\.|,)',  # According to X
        ]

    async def process_for_llm(self, raw_content: str, metadata: Dict, query: str = "") -> Dict[str, Any]:
        """Main processing pipeline for LLM optimization"""
        if not raw_content or len(raw_content) < self.min_content_length:
            return self._empty_response()
        
        # Clean and normalize content
        cleaned_content = self._clean_content(raw_content)
        
        # Extract different content types
        code_snippets = self._extract_code_snippets(cleaned_content)
        tables = self._extract_tables(cleaned_content)
        key_facts = self._extract_key_facts(cleaned_content, query)
        citations = self._extract_citations(cleaned_content)
        
        # Generate summary if content is too long
        summary = ""
        if len(cleaned_content) > 2000:
            summary = self._create_summary(cleaned_content, query)
        
        # Calculate content metrics
        reading_time = self._estimate_reading_time(cleaned_content)
        complexity_score = self._calculate_complexity(cleaned_content)
        content_type = self._detect_content_type(cleaned_content, metadata)
        confidence_score = self._calculate_confidence_score(cleaned_content, metadata)
        
        # Format final content for LLM
        formatted_content = self._format_for_llm_readability(cleaned_content)
        
        return {
            "full_content": formatted_content,
            "summary": summary,
            "key_facts": key_facts,
            "code_snippets": code_snippets,
            "tables": tables,
            "citations": citations,
            "metadata": {
                **metadata,
                "reading_time_minutes": reading_time,
                "complexity_score": complexity_score,
                "content_type": content_type,
                "confidence_score": confidence_score,
                "content_length": len(cleaned_content),
                "processed_at": datetime.utcnow().isoformat()
            }
        }

    def _clean_content(self, content: str) -> str:
        """Clean and normalize content for better processing"""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        
        # Remove common noise patterns
        noise_patterns = [
            r'Share on (Facebook|Twitter|LinkedIn)',
            r'Follow us on .+',
            r'Subscribe to .+',
            r'Advertisement',
            r'Related Articles?:?',
            r'Tags?:.*',
            r'Categories?:.*'
        ]
        
        for pattern in noise_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content.strip()

    def _extract_code_snippets(self, content: str) -> List[Dict[str, str]]:
        """Extract and format code snippets"""
        code_snippets = []
        
        for pattern, code_type in self.code_patterns:
            matches = re.finditer(pattern, content, re.DOTALL | re.MULTILINE)
            for match in matches:
                if code_type == 'code_block':
                    language = match.group(1) or 'unknown'
                    code = match.group(2).strip()
                else:
                    language = 'unknown'
                    code = match.group(1).strip()
                
                if len(code) > 10:  # Skip very short snippets
                    code_snippets.append({
                        "language": language,
                        "code": code,
                        "type": code_type
                    })
        
        return code_snippets[:10]  # Limit to prevent overwhelming

    def _extract_tables(self, content: str) -> List[Dict[str, Any]]:
        """Extract and structure table data"""
        tables = []
        
        # Try to find HTML tables
        soup = BeautifulSoup(content, 'html.parser')
        html_tables = soup.find_all('table')
        
        for table in html_tables[:5]:  # Limit to 5 tables
            headers = []
            rows = []
            
            # Extract headers
            header_row = table.find('tr')
            if header_row:
                headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
            
            # Extract data rows
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = [td.get_text().strip() for td in row.find_all(['td', 'th'])]
                if cells:
                    rows.append(cells)
            
            if headers and rows:
                tables.append({
                    "headers": headers,
                    "rows": rows[:10],  # Limit rows
                    "markdown": self._table_to_markdown(headers, rows[:10])
                })
        
        return tables

    def _extract_key_facts(self, content: str, query: str = "") -> List[str]:
        """Extract key facts and important information"""
        key_facts = []
        
        # Extract sentences with fact indicators
        for pattern in self.fact_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            key_facts.extend(matches)
        
        # Extract sentences containing query terms
        if query:
            query_words = query.lower().split()
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if (len(sentence) > 50 and 
                    any(word in sentence.lower() for word in query_words)):
                    key_facts.append(sentence + '.')
        
        # Extract list items
        list_items = re.findall(r'^[-*•]\s+(.+)$', content, re.MULTILINE)
        key_facts.extend(list_items)
        
        # Clean and deduplicate
        key_facts = [fact.strip() for fact in key_facts if len(fact.strip()) > 20]
        key_facts = list(dict.fromkeys(key_facts))  # Remove duplicates
        
        return key_facts[:self.max_key_points]

    def _extract_citations(self, content: str) -> List[str]:
        """Extract citations and references"""
        citations = []
        
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, content)
            citations.extend(matches)
        
        # Clean and deduplicate
        citations = [cite.strip() for cite in citations if len(cite.strip()) > 5]
        citations = list(dict.fromkeys(citations))
        
        return citations[:20]

    def _create_summary(self, content: str, query: str = "") -> str:
        """Create an intelligent summary of the content"""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
        
        if not sentences:
            return ""
        
        # Score sentences based on relevance
        scored_sentences = []
        query_words = query.lower().split() if query else []
        
        for sentence in sentences:
            score = 0
            sentence_lower = sentence.lower()
            
            # Query relevance
            for word in query_words:
                if word in sentence_lower:
                    score += 2
            
            # Position bonus (first and last sentences often important)
            if sentence == sentences[0]:
                score += 1
            if sentence == sentences[-1]:
                score += 1
            
            # Length normalization (prefer medium-length sentences)
            if 50 <= len(sentence) <= 200:
                score += 1
            
            # Key phrase indicators
            key_phrases = ['important', 'key', 'main', 'primary', 'significant', 'crucial']
            if any(phrase in sentence_lower for phrase in key_phrases):
                score += 1
            
            scored_sentences.append((score, sentence))
        
        # Sort by score and select top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        summary_sentences = []
        current_length = 0
        
        for score, sentence in scored_sentences:
            if current_length + len(sentence) <= self.max_summary_length:
                summary_sentences.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        return ' '.join(summary_sentences)

    def _estimate_reading_time(self, content: str) -> int:
        """Estimate reading time in minutes"""
        word_count = len(content.split())
        # Average reading speed: 200-250 words per minute
        return max(1, round(word_count / 225))

    def _calculate_complexity(self, content: str) -> float:
        """Calculate content complexity score (0-1)"""
        try:
            # Simple complexity calculation based on sentence and word length
            words = content.split()
            sentences = content.split('.')
            
            if not words or not sentences:
                return 0.5
            
            avg_words_per_sentence = len(words) / max(1, len(sentences))
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Normalize to 0-1 scale
            sentence_complexity = min(1.0, avg_words_per_sentence / 25)  # 25+ words = complex
            word_complexity = min(1.0, (avg_word_length - 3) / 7)  # 10+ chars = complex
            
            complexity = (sentence_complexity + word_complexity) / 2
            return round(max(0, min(1, complexity)), 2)
        except:
            return 0.5

    def _detect_content_type(self, content: str, metadata: Dict) -> str:
        """Detect the type of content"""
        content_lower = content.lower()
        
        # Check for code content
        if (any(lang in content_lower for lang in ['python', 'javascript', 'java', 'html', 'css']) or
            len(re.findall(r'def |function |class |import |#include', content)) > 3):
            return 'tutorial_code'
        
        # Check for academic content
        if (any(word in content_lower for word in ['abstract', 'methodology', 'conclusion', 'references']) or
            len(re.findall(r'\([A-Z][a-z]+ et al\., \d{4}\)', content)) > 2):
            return 'academic'
        
        # Check for news content
        if (metadata.get('publish_date') and 
            any(word in content_lower for word in ['breaking', 'reported', 'announced', 'sources say'])):
            return 'news'
        
        # Check for how-to content
        if (any(phrase in content_lower for phrase in ['step 1', 'first,', 'tutorial', 'how to', 'guide']) or
            len(re.findall(r'^\d+\.', content, re.MULTILINE)) > 3):
            return 'how_to'
        
        # Check for documentation
        if any(word in content_lower for word in ['api', 'documentation', 'reference', 'parameters']):
            return 'documentation'
        
        return 'article'

    def _calculate_confidence_score(self, content: str, metadata: Dict) -> float:
        """Calculate confidence in content quality (0-1)"""
        score = 0.5  # Base score
        
        # Content length (optimal range)
        length = len(content)
        if 500 <= length <= 10000:
            score += 0.2
        elif length > 100:
            score += 0.1
        
        # Has author
        if metadata.get('author'):
            score += 0.1
        
        # Has publish date
        if metadata.get('publish_date'):
            score += 0.1
        
        # Content structure (headings, lists)
        if len(re.findall(r'^#+\s', content, re.MULTILINE)) > 0:
            score += 0.1
        
        # No excessive advertisements or noise
        noise_indicators = ['click here', 'advertisement', 'sponsored']
        if not any(noise in content.lower() for noise in noise_indicators):
            score += 0.1
        
        return min(1.0, score)

    def _format_for_llm_readability(self, content: str) -> str:
        """Format content for optimal LLM readability"""
        # Ensure proper markdown formatting
        lines = content.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append('')
                continue
            
            # Format headings
            if line.isupper() and len(line) > 5:
                line = f"## {line.title()}"
            elif line.endswith(':') and len(line) < 100:
                line = f"**{line}**"
            
            formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)

    def _table_to_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        """Convert table data to markdown format"""
        if not headers or not rows:
            return ""
        
        # Create header row
        md_lines = ['| ' + ' | '.join(headers) + ' |']
        md_lines.append('|' + '---|' * len(headers))
        
        # Add data rows
        for row in rows:
            # Pad row to match header length
            padded_row = row + [''] * (len(headers) - len(row))
            md_lines.append('| ' + ' | '.join(padded_row[:len(headers)]) + ' |')
        
        return '\n'.join(md_lines)

    def _empty_response(self) -> Dict[str, Any]:
        """Return empty response structure"""
        return {
            "full_content": "",
            "summary": "",
            "key_facts": [],
            "code_snippets": [],
            "tables": [],
            "citations": [],
            "metadata": {
                "reading_time_minutes": 0,
                "complexity_score": 0,
                "content_type": "unknown",
                "confidence_score": 0,
                "content_length": 0,
                "processed_at": datetime.utcnow().isoformat()
            }
        }