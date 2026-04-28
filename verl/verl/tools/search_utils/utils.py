import uuid
import time
import os
import signal
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
try:
    from ddgs import DDGS
except ImportError:
    DDGS = None
# from crawl4ai import *

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
BOCHA_API_KEY = os.environ.get("BOCHA_API_KEY", "")


def tavily_search(queries, top_k=5):
    """Search using Tavily API — accessible from AutoDL containers."""
    if not TAVILY_API_KEY:
        return []
    query = queries[0] if queries else ""
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": TAVILY_API_KEY, "query": query, "max_results": top_k},
            timeout=15,
        )
        data = resp.json()
        results = data.get("results", [])
        return [
            {"query": query, "title": r.get("title", ""), "body": r.get("content", ""), "href": r.get("url", "")}
            for r in results
        ]
    except Exception:
        return []


def bocha_search(queries, top_k=5):
    """Search using Bocha API — domestic Chinese search service."""
    if not BOCHA_API_KEY:
        return []
    query = queries[0] if queries else ""
    for attempt in range(2):
        try:
            resp = requests.post(
                "https://api.bochaai.com/v1/web-search",
                headers={"Authorization": f"Bearer {BOCHA_API_KEY}", "Content-Type": "application/json"},
                json={"query": query, "count": top_k, "freshness": "noLimit", "summary": True},
                timeout=20,
            )
            if resp.status_code == 429:
                print(f"[bocha_search] Rate limited (429), attempt {attempt+1}/2, retrying after 5s...")
                time.sleep(5)
                continue
            if resp.status_code != 200:
                print(f"[bocha_search] HTTP {resp.status_code}: {resp.text[:200]}")
                return []
            data = resp.json()
            web_pages = data.get("data", {}).get("webPages", {}).get("value", [])
            return [
                {"query": query, "title": r.get("name", ""), "body": r.get("snippet", ""), "href": r.get("url", "")}
                for r in web_pages
            ]
        except Exception as e:
            print(f"[bocha_search] Exception attempt {attempt+1}/2: {e}")
            if attempt < 1:
                time.sleep(2)
    return []


def generate_snippet_id() -> str:
    """
    结合 UUID 的唯一性和 Base62 的字符集。
    生成格式如: 'S_BP4aUVA'
    """
    # 1. 定义字符集 (62个字符)
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # 2. 生成一个随机的 UUID
    # uuid4().int 会得到一个非常大的整数
    u_int = uuid.uuid4().int
    
    # 3. 将这个大整数转换为 62 进制字符串
    # 我们只需要取其中一段，这里通过取模和连续除法获得
    res = []
    # 循环 8 次生成 8 位字符（根据你的示例 S_BP4aUVA，BP4aUVA是7位，你可以改循环次数）
    for _ in range(7):
        u_int, remainder = divmod(u_int, 62)
        res.append(alphabet[remainder])
    
    return "S_" + "".join(res)

    
def generate_search_snippets(results):
    """
    results: [{"query":, "", "title": "", "body": "", "href": ""} ... ]
    Return:
    <snippet id="S_BP4aUVA">
    Title: xxx
    URL: https://cs.bjut.edu.cn/info/1509/3619.htm
    Text: xxx
    </snippet>
    """
    if not isinstance(results, list) or len(results) == 0:
        return "Google search encountered an error and was unable to extract valid information."
    
    result_text = ""
    for item in results:
        if not isinstance(item, dict):
            continue
        
        snippet_id = generate_snippet_id()
        start_str = "<snippet id=" + generate_snippet_id() + ">\n"
        end_str = "\n</snippet>"
        content = "Title: " + item.get("title", "") + "\n" + "URL: " + item.get("href", "") + "\n" + "Text: " + item.get("body", "")
        result_text += (start_str + content + end_str + "\n")
    
    return result_text.strip()

    
def ddgs_search(queries, top_k=5, ddgs_backend='auto'):
    if DDGS is None:
        return []
    def ddgs_single_search(query):
        for i in range(1):
            with DDGS(timeout=5) as _ddgs:
                try:
                    single_result = _ddgs.text(
                        query,
                        region="us-en",
                        safesearch="off",
                        max_results=10,
                        timelimit=None,
                        ddgs_backend=ddgs_backend
                    )
                    single_result = [{'query': query, **x} for x in single_result]
                    return single_result
                except Exception as e:
                    time.sleep(1)
                    continue
        return []

    results = []
    
    try:
        # 用线程池强制20秒超时，防止DDGS连接建立后永久挂起
        # 注意：shutdown(wait=False) 确保超时后不等待卡住的线程
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(ddgs_single_search, queries[0])
        try:
            result = future.result(timeout=20)
        except FuturesTimeoutError:
            result = []
        finally:
            executor.shutdown(wait=False)
        if result:
            return result[:top_k]
    except:
        pass
    
    return []