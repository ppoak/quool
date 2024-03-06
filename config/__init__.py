import quool


qtd = quool.PanelTable("./data/quotes-day", code_level="order_book_id", date_level="date")
qtm = quool.PanelTable("./data/quotes-min", code_level="order_book_id", date_level="datetime")
fqtd = quool.Factor("./data/quotes-day", code_level="order_book_id", date_level="date")
fqtm = quool.Factor("./data/quotes-min", code_level="order_book_id", date_level="date")
prx = quool.ProxyManager("./data/proxy")
