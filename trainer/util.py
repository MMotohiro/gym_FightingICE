from functools import wraps
import time

def stop_watch(func):
    """
    関数の実行時間を計測する.
    参考文献(丸パクリ)
    https://qiita.com/hisatoshi/items/7354c76a4412dffc4fd7
    """
    @wraps(func)
    def wrapper(*args, **kargs) :
        start = time.time()
        result = func(*args,**kargs)
        elapsed_time =  time.time() - start
        print(f"{func.__name__}は{elapsed_time}秒かかりました")
        return result
    return wrapper