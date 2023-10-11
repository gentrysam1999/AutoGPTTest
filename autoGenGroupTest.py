import autogen
from dotenv import load_dotenv
from os import environ


# Config Setup
load_dotenv()
config_list_dumb = [{'model': 'gpt-3.5-turbo', 'api_key': environ.get('API_KEY')}]
config_list_smart = [{'model': 'gpt-4', 'api_key': environ.get('API_KEY')}]
llm_config_dumb = {
    "request_timeout": 600,
    "seed": 42,
    "config_list": config_list_dumb,
    "temperature": 0
}
llm_config_smart = {
    "request_timeout": 600,
    "seed": 42,
    "config_list": config_list_smart,
    "temperature": 0
}


def runAutoGenTest():
    # Agent Setup
    engineer = autogen.AssistantAgent(
        name="Engineer",
        llm_config=llm_config_dumb,
        system_message='''Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
        Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
        If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
        ''',
    )
    planner = autogen.AssistantAgent(
        name="Planner",
        system_message='''Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
        The plan will involve an engineer who can write code.
        Explain the plan first. Be clear which step is performed by an engineer.
        ''',
        llm_config=llm_config_dumb,
    )
    executor = autogen.UserProxyAgent(
        name="Executor",
        system_message="Executor. Execute the code written by the engineer and report the result.",
        human_input_mode="NEVER",
        code_execution_config={"last_n_messages": 3, "work_dir": "web"},
    )
    critic = autogen.AssistantAgent(
        name="Critic",
        system_message="Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.",
        llm_config=llm_config_dumb,
    )

    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        human_input_mode="TERMINATE",
        max_consecutive_auto_reply=1,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={"work_dir": "web"},
        llm_config=llm_config_dumb,
        system_message="""A human admin"""
    )

    groupchat = autogen.GroupChat(agents=[user_proxy, engineer, planner, executor, critic], messages=[], max_round=50)
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config_dumb)

    # TASKS
    task = """
    Write python code to output numbers 1 to 100, and then store the code in a file. Do not use any other language than python.
    """

    user_proxy.initiate_chat(
        manager,
        message=task
    )



if __name__ == "__main__":
    load_dotenv()
    runAutoGenTest()
    
