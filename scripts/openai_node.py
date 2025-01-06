#!/usr/bin/env python3

import rospy
from openai import AzureOpenAI, OpenAI

from openai_ros.msg import StringArray
from openai_ros.srv import Completion, CompletionResponse, Embedding, EmbeddingResponse

API_TYPES = [
    "completion",
    "embedding",
]


def servicer_completion(req):
    global client, max_tokens, model, api_type, default_request_timeout, enable_chat
    res = CompletionResponse()

    if enable_chat:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": req.prompt,
                }
            ],
            temperature=req.temperature,
            max_tokens=max_tokens,
            stop=req.stop,
        )
        print(response)
        res.finish_reason = response.choices[0].finish_reason
        res.text = response.choices[0].message.content
        res.model = response.model
        res.completion_tokens = response.usage.completion_tokens
        res.prompt_tokens = response.usage.prompt_tokens
        res.total_tokens = response.usage
    else:
        response = client.completions.create(
            model=model,
            prompt=req.prompt,
            temperature=req.temperature,
            max_tokens=max_tokens,
            stop=req.stop,
        )
        res.finish_reason = response.choices[0].finish_reason
        res.text = response.choices[0].text
        res.model = response.model
        res.completion_tokens = response.usage.completion_tokens
        res.prompt_tokens = response.usage.prompt_tokens
        res.total_tokens = response.usage.total_tokens

    rospy.loginfo(f"req: {req}, res:{res}")

    # When response is not working, completion_tokens is None, which chase error on CompleteResponse format(int32)
    if not isinstance(res.completion_tokens, int):
        res.completion_tokens = -1
    return res


def servicer_embedding(req):
    global client, model, api_type, default_request_timeout
    res = EmbeddingResponse()

    response = client.embeddings.create(
        model=model,
        input=[req.prompt],
        request_timeout=(
            req.timeout.to_sec()
            if req.timeout.to_sec() != 0.0
            else default_request_timeout
        ),
    )

    res.embedding = response.data[0].embedding
    res.model = response.model
    res.prompt_tokens = response.usage.prompt_tokens
    res.total_tokens = response.usage.total_tokens
    rospy.loginfo(f"req: {req}, res:{res}")
    return res


def main():
    global client, max_tokens, model, api_type, default_request_timeout, enable_chat
    pub = rospy.Publisher("available_models", StringArray, queue_size=1, latch=True)
    rospy.init_node("openai_node", anonymous=True)
    use_azure = rospy.get_param("~use_azure", default=False)
    enable_chat = rospy.get_param("~enable_chat", default=False)

    if use_azure:
        client = AzureOpenAI(
            api_key=rospy.get_param("~key"),
            azure_endpoint=rospy.get_param("~azure_endpoint"),
            api_version=rospy.get_param("~azure_api_version"),
        )
    else:
        client = OpenAI(api_key=rospy.get_param("~key"))
    max_tokens = rospy.get_param("~max_tokens", default=256)
    model = rospy.get_param("~model", default="text-davinci-003")
    api_type = rospy.get_param("~api_type", default="completion")
    default_request_timeout = rospy.get_param("~default_request_timeout", 10.0)

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
