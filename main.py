from Credentials import *
from Constants import *

import functional

import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from vk_api.vk_api import VkApiMethod

vk_session = vk_api.VkApi(token=token)


longpoll = VkBotLongPoll(vk_session, group_id)
meth = VkApiMethod(vk_session)


def pool():
    for i, event in enumerate(longpoll.listen()):
        print(event)
        if event.type == VkBotEventType.MESSAGE_NEW:
            functional.handle_msg(event)


if __name__ == '__main__':
    pool()
