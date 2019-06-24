from Constants import *
from main import meth


def handle_msg(event):
    text = event.obj['text'].lower()
    if event.obj['from_id'] == sar_id:
        if '?' in text:
            question(event.chat_id)
        else:
            msg = 'Саранж пошол нахой!!!!'
            write_msg(chat_id=event.chat_id, message=msg)
    elif 'санджар' in text:
        sandjar_mentioned(event.chat_id)


def question(chat_id):
    msg = 'Санджар, как же ты доебал со своими вопросами блядскими'
    write_msg(chat_id=chat_id, message=msg)


def sandjar_mentioned(chat_id):
    msg = 'вместо санджара теперь я конченый в этой конфе'
    write_msg(chat_id=chat_id, message=msg)


def write_msg(chat_id, message):
    meth.messages.send(chat_id=chat_id, message=message, random_id=bigint)