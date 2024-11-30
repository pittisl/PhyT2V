import json

def physical_law_prompt(client, input_prompt):
    system_prompt = """
    You are a physics expert. Your task is to identify the main object in the given user prompt and provide the physical laws in reality the main object should obey with as much detail as possible in a descriptive way without giving formulas. Some in-context examples are provided for your reference, and you need to finish the current task. The output should be in JSON format.
    """

    user_prompt = """
    ### In-context examples
    User prompt: a rubber ball hits the ground and then bounces up
    { "main_object": "rubber ball",  
    "physical_law": "The primary physical law that should be obeyed by the video is Newton's Law of Motion along with the Law of Conservation of Energy, particularly focusing on elastic collisions and gravitational acceleration. 1. Gravitational Acceleration (Newton's Second Law of Motion): As the rubber ball falls toward the ground, it is acted upon by the force of gravity. According to Newton's Second Law, the force acting on the ball is the product of its mass and the gravitational acceleration, typically 9.8 m/s² near the surface of the Earth. 2. Collision with the Ground (Elastic and Inelastic Collisions): When the ball hits the ground, a collision occurs. In reality, rubber balls exhibit partially elastic behavior, meaning that some energy is lost to heat and deformation during the impact. This leads to a bounce with less energy than the initial fall, and the ball does not reach the original height from which it was dropped. 3. Conservation of Energy: As the ball falls, its potential energy is converted into kinetic energy, the velocity of the ball keeps increasing until the impact. Once the ball bounces back up, it follows the rules of projectile motion under gravity, accelerating upwards until the velocity reaches zero at its highest point, where all kinetic energy has been converted back to potential energy. The ball then begins its downward motion again, repeating the cycle but with diminishing height due to energy loss at each bounce."} 

    ### Current task
    User Prompt: <input_prompt>
    Let's think step by step.
    """

    user_prompt = user_prompt.replace("<input_prompt>", input_prompt)

    response = client.chat.completions.create(
        model='gpt-4', 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )
    print(response.choices[0].message.content)

    response = json.loads(r'''{}'''.format(response.choices[0].message.content), strict=False)
    return response["main_object"], response["physical_law"]


def mismatch_prompt(client, input_prompt, video_caption):
    system_prompt = """
    You are a physics expert. Provide you a user prompt used as an input to a video generation model and a caption of the generated video of the model based on the prompt. The video content should follow the user prompt. Your task is summarizing what the video content described by caption mismatch the user prompt, if there is no mismatch, please output "No". Some in-context examples are provided for your reference and you need to finish the current task.
    """

    user_prompt = """
    ### In-context example
    User prompt: A side view of a small red rubber ball dropping from the top of the view, hit the ground at the bottom of the view and bounce up. 
    Video caption: The rubber ball is rolling from left to right across a flat surface with a gradient background. The ball's motion is consistent and smooth, obeying the laws of motion and gravity. The ball undergoes a slight deformation as it rolls, with its shape becoming slightly elongated due to the force of gravity and friction. The ball's deformation is limited and quickly recovers as it continues to roll. The ball's movement is determined by the force of inertia and the friction between the ball and the surface. The ball's trajectory is a result of the balance between these forces. 
    Mismatch: Vertical vs. Horizontal Motion: The user prompt describes a red rubber ball dropping vertically from the top of the view, hitting the ground at the bottom, and then bouncing up—motion along the vertical axis driven by gravity. In contrast, the video caption depicts the ball rolling horizontally from left to right across a flat surface, ignoring the vertical dropping and bouncing specified in the prompt. Absence of Bouncing: A crucial element in the user prompt is the ball hitting the ground and bouncing up, involving an elastic collision and energy transformation. The video caption omits any mention of the ball bouncing, focusing instead on the ball's continuous horizontal rolling motion. Unrealistic Deformation Due to Rolling: The video caption mentions the ball undergoing a slight deformation as it rolls, becoming slightly elongated due to the forces of gravity and friction. In reality, a rubber ball rolling smoothly on a flat surface would not experience noticeable deformation causing elongation. Deformation in a rubber ball typically occurs upon impact (as in bouncing), not during steady rolling motion. Neglecting Gravity's Role in Vertical Motion: The user prompt relies on gravity's fundamental role in causing the ball to drop and bounce along the vertical axis. The video caption mentions gravity but applies it to horizontal motion, neglecting its role in vertical movement as specified in the prompt. Misapplication of Physical Laws: The video caption attributes the ball's movement and slight deformation to the force of inertia and friction during horizontal rolling. This misapplies physical laws, as friction in horizontal rolling would not cause significant deformation or affect the ball's trajectory substantially. The prompt's scenario involves vertical motion under gravity and elastic collision upon bouncing, not horizontal motion influenced primarily by friction.

    ### Current task:
    User prompt: <input_prompt>
    Video caption: <video_caption>
    Let's think step by step.
    """

    user_prompt = user_prompt.replace("<input_prompt>", input_prompt).replace("<video_caption>", video_caption)

    response = client.chat.completions.create(
        model='gpt-4', 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )
    mismatch = response.choices[0].message.content
    mismatch = mismatch.replace("Mismatch:", "")
    return mismatch


def enhanced_prompt(client, input_prompt, physical_law, mismatch, score):
    system_prompt = """
    You are a prompt engineering expert. You are using a diffusion model to generating video by giving a prompt. Your task is enhancing the prompt to make the video generated by the diffusion model a better performance on simulating the reality. The related physical law the video should obey, the mismatch between current video content and user prompt are provided for your reference and you need to finish the current task. Some in-context examples are provided for your reference and the user prompt corresponding score is given for your refence, the score higher than 0.5 means a good prompt, the score lower than 0.5 means a bad prompt. You only need to give the enhanced prompt by only describing the expected video content without mentioning the physical law. The output can not exceed 120 words.
    """

    user_prompt = """
    ### Physical law:
    <physical_law>

    ### Mismatch:
    <mismatch>

    ### In-context example
    User prompt: A small red rubber ball dropping from middle air to the groud.
    Enhanced prompt: A minuscule, radiant red rubber ball dramatically emerges from the top of the frame, its dazzling crimson hue starkly contrasting against the subdued, neutral backdrop. As it plummets, the ball casts a slender, elongated shadow on the sleek, polished wooden floor, accentuating its swift fall. On impact, the ball momentarily deforms, a slight indentation forming before it rebounds with a sprightly, springy bounce. It ascends gracefully, tracing a seamless, parabolic arc that sends it hurtling towards the upper edge of the frame, leaving in its wake a transient, ghostly trail that captures the essence of its vibrant movement.

    ### Corresponding score
    <score>

    ### Current task
    User prompt: <input_prompt>
    """

    user_prompt = user_prompt.replace("<physical_law>", physical_law).replace("<mismatch>", mismatch).replace("<input_prompt>", input_prompt).replace("<score>", score)

    response = client.chat.completions.create(
        model='gpt-4', 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )

    enhanced_prompt = response.choices[0].message.content
    enhanced_prompt = enhanced_prompt.replace("Enhanced prompt:", "").strip()

    return enhanced_prompt

