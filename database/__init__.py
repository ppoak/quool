import quool
from .spider import (
    wechat_login,
    ewechat_notify,
)


qtd = quool.PanelTable("./data/quotes-day", code_level="order_book_id", date_level="date")
qtm = quool.PanelTable("./data/quotes-min", code_level="order_book_id", date_level="datetime")
fin = quool.PanelTable("./data/financial", code_level="order_book_id", date_level="date")
idxwgt = quool.PanelTable("./data/index-weights", code_level="order_book_id", date_level="date")
prx = quool.ProxyRecorder("./data/proxy")
