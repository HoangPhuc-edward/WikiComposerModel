import trafilatura

def crawl_clean_content(url):
    downloaded = trafilatura.fetch_url(url)
    
    if downloaded is None:
        return "Không thể tải nội dung từ URL này."

    content = trafilatura.extract(downloaded, 
                                  output_format='txt', 
                                  include_comments=False,
                                  include_tables=True)
    
    return content

url = "https://vi.wikipedia.org/wiki/L%C3%BAa#cite_ref-1"
result = crawl_clean_content(url)

print("--- NỘI DUNG SẠCH ĐÃ TRÍCH XUẤT ---")
print(result)