#!/usr/bin/env python3

import rospy
from openai import OpenAI

from openai_ros.msg import StringArray
from openai_ros.srv import Completion, CompletionResponse, Embedding, EmbeddingResponse

API_TYPES = [
    "completion",
    "embedding",
]


def servicer_completion(req):
    global client, max_tokens, model, api_type
    res = CompletionResponse()

    response = client.completions.create(
        model=model,
        prompt=req.prompt,
        temperature=req.temperature,
        max_tokens=max_tokens,
    )

    res.finish_reason = response.choices[0].finish_reason
    res.text = response.choices[0].text
    res.model = response.model
    res.completion_tokens = response.usage.completion_tokens
    res.prompt_tokens = response.usage.prompt_tokens
    res.total_tokens = response.usage.total_tokens

    # When response is not working, completion_tokens is None, which chase error on CompleteResponse format(int32)
    if not isinstance(res.completion_tokens, int):
        res.completion_tokens = -1
    return res


def servicer_embedding(req):
    global client, model, api_type
    res = EmbeddingResponse()

    response = client.embeddings.create(
        model=model,
        inputs=[req.prompt],
    )

    res.embeddings = response.data
    res.model = response.model
    res.prompt_tokens = response.usage.prompt_tokens
    res.total_tokens = response.usage.total_tokens
    return res


def main():
    global client, max_tokens, model, api_type
    pub = rospy.Publisher("available_models", StringArray, queue_size=1, latch=True)
    rospy.init_node("openai_node", anonymous=True)

    client = OpenAI(api_key=rospy.get_param("~key"))
    max_tokens = rospy.get_param("~max_tokens", default=256)
    model = rospy.get_param("~model", default="text-davinci-003")
    api_type = rospy.get_param("~api_type", default="completion")

    if api_type not in API_TYPES:
        rospy.logwarn(api_type + " is not an available API type")
        rospy.logwarn(API_TYPES)
        return

    models_msg = StringArray()
    for m in client.models.list():
        models_msg.data.append(m.id)

    if model not in models_msg.data:
        rospy.logwarn(model + " is not an available model")
        rospy.logwarn(models_msg.data)
        return

    pub.publish(models_msg)

    if api_type == "completion":
        rospy.logwarn("API Type: Completion")
        rospy.Service("get_response", Completion, servicer_completion)
    elif api_type == "embedding":
        rospy.logwarn("API Type: Embedding")
        rospy.Service("get_embedding", Embedding, servicer_embedding)
    else:
        rospy.logwarn(api_type + " is not an available API type")
        return
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
