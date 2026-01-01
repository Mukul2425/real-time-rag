# Contributing to Multimodal EV RAG Assistant

Thank you for your interest in contributing to the Multimodal EV RAG Assistant! This document provides guidelines and information for contributors.

## 🎯 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear descriptive title**
- **Detailed steps to reproduce**
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, GPU/CPU)
- **Relevant logs or error messages**
- **Screenshots** if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When suggesting an enhancement:

- **Use a clear and descriptive title**
- **Provide detailed description** of the proposed functionality
- **Explain why this enhancement would be useful**
- **List any potential drawbacks or challenges**

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Write clear commit messages**
6. **Submit a pull request**

## 🛠️ Development Setup

### Local Environment

```bash
# Clone your fork
git clone https://github.com/<YOUR_USERNAME>/real-time-rag.git
cd real-time-rag

# Add upstream remote
git remote add upstream https://github.com/<ORIGINAL_OWNER>/real-time-rag.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies (if we add them)
# pip install -r requirements-dev.txt
```

### Environment Variables

Create a `.env` file with your API keys:
```env
NEWS_API_KEY=your_test_key
PINECONE_API_KEY=your_test_key
OPENROUTER_API_KEY=your_test_key
```

**Important**: Never commit your `.env` file!

## 📝 Code Style Guidelines

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (120 max for complex lines)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings, single for dict keys
- **Imports**: Grouped and sorted (standard library, third-party, local)

Example:
```python
import os
import sys

import requests
from dotenv import load_dotenv

from local_module import helper
```

### Naming Conventions

- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Documentation

- Add docstrings to all functions and classes
- Use clear, descriptive variable names
- Comment complex logic
- Update README.md for user-facing changes

Example docstring:
```python
def create_query_embedding(query_text, query_image=None):
    """
    Create embedding for text and/or image query using CLIP.
    
    Args:
        query_text (str): The text query to embed
        query_image (PIL.Image, optional): Image to embed with text
        
    Returns:
        list: Embedding vector of dimension 512
        
    Raises:
        ValueError: If query_text is empty
    """
    pass
```

## 🧪 Testing

### Manual Testing

Before submitting a PR, test:

1. **Text-only queries** work correctly
2. **Image upload** functionality works
3. **Multimodal queries** (text + image) work
4. **Error handling** for edge cases
5. **Different data sources** if modified

### Testing Checklist

- [ ] Code runs without errors
- [ ] All existing functionality still works
- [ ] New features work as expected
- [ ] Error messages are clear and helpful
- [ ] Performance is acceptable
- [ ] Documentation is updated

## 🔍 Code Review Process

### What We Look For

- **Correctness**: Does the code work as intended?
- **Quality**: Is the code clean, readable, and well-structured?
- **Testing**: Has it been tested adequately?
- **Documentation**: Are changes documented?
- **Security**: Does it introduce any vulnerabilities?
- **Performance**: Is it reasonably efficient?

### Review Timeline

- Initial review within 1-3 days
- Follow-up reviews within 1-2 days
- Merging after approval from maintainer

## 🎨 Areas for Contribution

### High Priority

- [ ] Add comprehensive test suite
- [ ] Improve error handling and logging
- [ ] Optimize image processing performance
- [ ] Add monitoring and metrics dashboard
- [ ] Implement caching layer

### Medium Priority

- [ ] Support for additional data sources
- [ ] Multi-language support
- [ ] Advanced filtering options
- [ ] Export functionality for results
- [ ] User authentication system

### Good First Issues

- [ ] Improve documentation with more examples
- [ ] Add more detailed error messages
- [ ] Create utility scripts for common tasks
- [ ] Add validation for configuration
- [ ] Improve UI/UX of Streamlit app

## 🚀 Suggested Improvements

### Code Quality

**For comprehensive code improvement suggestions with examples, see [CODE_IMPROVEMENTS.md](../CODE_IMPROVEMENTS.md)**

Here's a quick summary:

1. **Add Type Hints**: Use Python type annotations throughout
   ```python
   def process_article(article: dict) -> List[dict]:
       pass
   ```

2. **Error Handling**: Improve try-except blocks with specific exceptions
   ```python
   try:
       result = process_data()
   except ConnectionError as e:
       logger.error(f"Connection failed: {e}")
       raise
   except Exception as e:
       logger.error(f"Unexpected error: {e}")
       return None
   ```

3. **Logging**: Add structured logging
   ```python
   import logging
   
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   
   logger.info(f"Processing article: {article_id}")
   ```

4. **Configuration Management**: Move hardcoded values to config
   ```python
   # config.py
   CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
   EMBEDDING_DIMENSION = 512
   MAX_IMAGE_SIZE_MB = 5
   ```

5. **Async Processing**: Consider async/await for I/O operations
   ```python
   async def download_image(url: str) -> Image:
       async with aiohttp.ClientSession() as session:
           async with session.get(url) as response:
               return await response.read()
   ```

### Architecture Improvements

1. **Decouple Components**: Separate concerns into modules
   - `embeddings.py` - Embedding generation
   - `storage.py` - Pinecone operations
   - `retrieval.py` - Search logic
   - `ui.py` - Streamlit interface

2. **Add Caching**: Cache embeddings and API responses
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def get_embedding(text: str) -> List[float]:
       pass
   ```

3. **Implement Retry Logic**: Add exponential backoff
   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential
   
   @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
   def api_call():
       pass
   ```

4. **Add Monitoring**: Implement metrics collection
   ```python
   from prometheus_client import Counter, Histogram
   
   articles_processed = Counter('articles_processed_total', 'Total articles processed')
   processing_time = Histogram('article_processing_seconds', 'Time to process article')
   ```

### Testing Improvements

1. **Unit Tests**: Add pytest-based tests
   ```python
   # tests/test_embeddings.py
   import pytest
   from embeddings import create_query_embedding
   
   def test_text_embedding():
       result = create_query_embedding("test query")
       assert len(result) == 512
       assert all(isinstance(x, float) for x in result)
   ```

2. **Integration Tests**: Test component interactions
3. **Mock External Services**: Don't call real APIs in tests
4. **Performance Tests**: Benchmark critical operations

## 📋 Commit Message Guidelines

Format: `<type>(<scope>): <subject>`

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(embeddings): Add support for custom CLIP models
fix(consumer): Handle corrupted image downloads gracefully
docs(readme): Update installation instructions
refactor(app): Simplify query processing logic
```

## 🔐 Security Guidelines

- **Never commit sensitive data** (API keys, passwords)
- **Validate user input** to prevent injection attacks
- **Use parameterized queries** for any database operations
- **Keep dependencies updated** to patch vulnerabilities
- **Review third-party libraries** before adding them

## 📞 Getting Help

- **Questions?** Open a GitHub Discussion
- **Bug?** Open an Issue with details
- **Feature idea?** Open an Issue with proposal
- **Need clarification?** Comment on existing issues/PRs

## 🙏 Recognition

Contributors will be recognized in:
- GitHub Contributors page
- Project README (for significant contributions)
- Release notes (for features/fixes in that release)

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to making this project better! 🚀**
