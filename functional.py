from numpy.random import binomial
from Constants import *
from main import meth


statistics = {'messages': 0, 'questions': 0, 'pictures': 0}


sar_nicknames = ['санджар', 'санжар', 'саранж', 'сарандж', 'саранжиан',
                 'саранджиан', 'санжарчик', 'санджарчик', 'тлепин', 'тверин']
dikiy_dog_names = {'емелю': ['эмеля', 'емеля', 'вова', 'володя', 'владимир', 'чернявский', 'черняуски'],
                    'даню': ['даниил', 'данил', 'даня', 'даник', 'красильников', 'красильник', 'данильников'],
                    'стаса': ['стас', 'станислав', 'стасян', 'стасянчик', 'стасянслав'],
                    'витю': ['витя', 'витек', 'виктор', 'витёк', 'павлишен'],
                    'иру': ['ира', 'ирина', 'петелниа', 'петелинка']}


def handle_msg(event):
    text = event.obj['text'].lower()

    statistics['messages'] += 1

    if event.obj['from_id'] == sar_id:
        if '?' in text:
            question(event.chat_id, text)
            statistics['questions'] += 1
        elif dikiy_dog_mentioned(text):
            for name in dikiy_dog_names:
                if any(nick in text for nick in dikiy_dog_names.get(name)):
                    msg = 'санджар, не доёбывай блять ' + name
                    write_msg(chat_id=event.chat_id, message=msg)

        elif binomial(1, 0.5, 1):
            msg = 'Саранж пошол нахой!!!!'
            write_msg(chat_id=event.chat_id, message=msg)

    elif any([nick in text for nick in sar_nicknames]):
        sar_mentioned(event.chat_id)
    elif text == 'statistics':
        '''тут написан говнокод'''
        msg = 'Охуеть Санджар сегодня написал' + str(statistics['messages']) + 'бесполезных сообщений\nЗадал' +\
              str(statistics['questions']) + ' ебучих вопроса\nи еще наделал много бесполезной хуйни'
        write_msg(chat_id=event.chat_id, message=msg)


def question(chat_id, text):
    if 'как' in text:
        msg = 'никак, отъебись со своими вопросами'
        write_msg(chat_id=chat_id, message=msg)
    if 'кто' in text:
        msg = 'хуй в пальто, отъебись блять'
        write_msg(chat_id=chat_id, message=msg)
    if 'где' in text:
        msg = 'в пизде, заебал со своими вопросами уже'
        write_msg(chat_id=chat_id, message=msg)
    if 'когда' in text:
        msg = 'когда рак на горе свиснет, ебнуться от твоих вопросов можно'
        write_msg(chat_id=chat_id, message=msg)
    if 'зачем' in text:
        msg = 'меня уже заебало писать логику, а тебя не заебало вопросы писать??'
        write_msg(chat_id=chat_id, message=msg)
    if 'почему' in text:
        msg = 'потому что ты тупой, как и твои вопросы блядские'
        write_msg(chat_id=chat_id, message=msg)
    if 'откуда' in text:
        msg = 'санджар залезь туда, откуда вылез'
        write_msg(chat_id=chat_id, message=msg)
    else:
        msg = 'санджар как же ты доебал со своими вопросами блядскими'
        write_msg(chat_id=chat_id, message=msg)


def sar_mentioned(chat_id):
    msg = 'вместо санджара теперь я конченый в этой конфе'
    write_msg(chat_id=chat_id, message=msg)


def write_msg(chat_id, message):
    meth.messages.send(chat_id=chat_id, message=message, random_id=bigint)


def dikiy_dog_mentioned(text):
    for name in dikiy_dog_names:
        if any([nick in text for nick in dikiy_dog_names.get(name)]):
            return True
    return False

