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
    _questions = ['как', 'кто', 'где', 'когда', 'зачем', 'почему', 'откуда']
    answers = {
        'как': 'никак, отъебись со своими вопросами',
        'кто': 'хуй в пальто, отъебись блять',
        'где': 'в пизде, заебал со своими вопросами уже',
        'когда': 'когда рак на горе свиснет, ебнуться от твоих вопросов можно',
        'зачем': 'меня уже заебало писать логику, а тебя не заебало вопросы писать??',
        'почему': 'потому что ты тупой, как и твои вопросы блядские',
        'откуда': 'санджар залезь туда, откуда вылез'
    }

    for q in _questions:
        if q in text:
            write_msg(chat_id=chat_id, message=answers[q])


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
