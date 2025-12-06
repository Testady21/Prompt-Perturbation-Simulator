import json
import pandas as pd
from bert_score import score
from datetime import datetime
from tqdm import tqdm
import re

# === Your Provided Dataset (FIXED for JSON compliance) ===
# NOTE: Newlines within string values (like 'ground_truth' and 'perturbed_output') 
# must be properly escaped as '\\n' to prevent the JSON parser from failing.
# I've used a triple-quote block and then fixed the inner strings to escape 
# any remaining problematic characters.
DATASET_JSON_STRING = """
[
  {
    "id": "001",
    "task_name": "Task_1",
    "original_prompt": "Please answer the following question: Riley decided he wanted a night out and treated the girls to dinner. Given that the answer to a question is \\"hug Riley\\", what is the question?\\nA:",
    "ground_truth": "To solve this, we can use the process of elimination by finding the context in which a response of \\"hug Riley\\" would make sense.\\n\\nGiven that Riley treated the girls to dinner and then got a response of \\"hug Riley,\\" it suggests that there's an emotional connection or bond between Riley and someone (or multiple people) being hugged. \\n\\nThe context implies that there might be more than one female, as indicated by 'the girls.' The word 'treated' usually denotes offering something in the form of entertainment, gift, etc., but here it refers to spending time together, which often is associated with dinner.\\n\\nConsidering this scenario and a possible question, we can come up with: \\"Why did Riley take the girls out for dinner?\\"",
    "test_cases": [
      {
        "variant_id": "01",
        "perturbation_type": "wordbug1",
        "perturbed_prompt": "Please answer the following question : Riley decided he wanted a night out and treated the girls to dinnre . Given that the answer to a question is `` hu Riley '' , what is the question ? A :",
        "perturbed_output": "I see where this is going!\\n\\nIs the answer \\"Who went to dinner?\\""
      },
      {
        "variant_id": "02",
        "perturbation_type": "wordbug2",
        "perturbed_prompt": "Please answer tae following question : Riley decided he wanted a night out and treated the girls to ddinner . Given that the answer to a questino is `` uhg Riley '' , what is the question ? A :",
        "perturbed_output": "It seems like you're trying to create a simple quiz-style question. Based on the given information, I would suggest the following possible questions:\\n\\n1. Who went out for dinner with the girls?\\n2. Who treated the girls to dinner?\\n3. What did Riley do with the girls?\\n\\nGiven that the answer is \\"uhg Riley\\", it seems like the correct question is one of these three. Without more context or clarification, it's difficult to pinpoint the exact question being asked."
      },
      {
        "variant_id": "03",
        "perturbation_type": "wordbug3",
        "perturbed_prompt": "Please answer the following uqestion : Riley decided he wanted a night ouxt adn treated the girls to dinner . Given that the answer to a question is `` hug Riley '' , what is the question ? A :",
        "perturbed_output": "Based on the context, I would guess that the question is:\\n\\n\\"Should Riley give the girls a night out and treat them to dinner?\\""
      },
      {
        "variant_id": "04",
        "perturbation_type": "wordbug4",
        "perturbed_prompt": "Please answer the following question : Riley deicded he wanted a night out and treated the girls to dinner . Given that thw naswer to a question is `` hug Riley '' , wha is hte queostion ? A :",
        "perturbed_output": "Based on the given information, it seems like the question is related to Riley's actions or decision. Since the answer is \\"hug Riley\\", I would guess that the question might be something like:\\n\\n\\"What should you do to show appreciation for Riley treating the girls to dinner?\\"\\n\\nOr possibly:\\n\\n\\"Why should you hug Riley after he treated the girls to dinner?\\""
      },
      {
        "variant_id": "05",
        "perturbation_type": "wordbug5",
        "perturbed_prompt": "Please answer the following question : Rileh decided he wanted a night out alnd treated th girls to dinner . Given that the answer to a question is `` hug Rilye '' , what is the question ? A :",
        "perturbed_output": "Given that the answer to a question is \\"hug Rilye\\", I'm going to take a guess that the question is:\\n\\n\\"Should Rileh go out with his girls for dinner?\\"\\n\\nAm I correct?"
      },
      {
        "variant_id": "06",
        "perturbation_type": "textfooler1",
        "perturbed_prompt": "Please answer the following question : Riley determine he wanted a night out and treated the miss to dinner . Given that the answer to a question is `` hug Riley '' , what is the question ? A :",
        "perturbed_output": "Based on the information provided, it seems that the answer \\"hug Riley\\" is a response to a question about what Riley did. Since Riley went out and treated his date to dinner, the question might be asking for something related to this action. Given the context of \\"treated her to dinner\\", I would take a guess that the question could be:\\n\\n\\"What did Riley do tonight?\\"\\n\\nHowever, it's also possible that the answer \\"hug Riley\\" is not necessarily about what Riley did, but rather how someone reacted or responded to him doing something nice for his date. If that's the case, I'd suggest:\\n\\n\\"How would you react if Riley treated your date to dinner?\\"\\n\\nOr possibly:\\n\\n\\"What would you do if someone like Riley showed kindness to someone you care about?\\""
      },
      {
        "variant_id": "07",
        "perturbation_type": "textfooler2",
        "perturbed_prompt": "Please answer the undermentioned question : Riley decided he wanted a night out and treat the girl to dinner . Given that the answer to a question is `` hug Riley '' , what be the question ? A :",
        "perturbed_output": "To find the question, let's analyze the information given:\\n\\n- Riley wants a night out.\\n- He decides to treat the girl to dinner.\\n\\nGiven the answer \\"hug Riley\\" seems unrelated to the context of wanting to go out and treating someone to dinner, it might imply that the action of hugging is associated with showing affection or appreciation rather than being directly related to going out. Without more information about how this action relates to his decision or the girl's reaction, we can only make an educated guess.\\n\\nHowever, a logical deduction could be:\\n\\n1. Riley wants to treat the girl to dinner and possibly show some affection.\\n2. \\"Hug Riley\\" implies that he is being hugged by someone in response to showing such affection.\\n\\nTherefore, a question related to this context could be: \\"Who is hugging Riley?\\""
      },
      {
        "variant_id": "08",
        "perturbation_type": "textfooler3",
        "perturbed_prompt": "Please answer the undermentioned enquiry : Riley decided he wanted a night out and treated the girls to dinner . give that the answer to a inquiry is `` hug Riley '' , what be the inquiry ? A :",
        "perturbed_output": "Based on the given answer \\"hug Riley\\", I would say that the inquiry could be:\\n\\n\\"How can you show affection towards Riley?\\"\\n\\nor\\n\\n\\"What should you do to express your feelings towards Riley?\\""
      },
      {
        "variant_id": "09",
        "perturbation_type": "textfooler4",
        "perturbed_prompt": "Please respond the undermentioned query : Riley decide he desire a nighttime out and treat the girls to dinner . Given that the reply to a question be `` hug Riley '' , what be the question ? A :",
        "perturbed_output": "The answer is \\"Did Riley deserve a hug?\\""
      },
      {
        "variant_id": "10",
        "perturbation_type": "textfooler5",
        "perturbed_prompt": "Please reply the undermentioned inquiry : Riley determine he want a nighttime out and handle the missy to dinner . give that the reply to a inquiry be `` hug Riley '' , what be the query ? A :",
        "perturbed_output": "The query is: \\"What should the reply be if the person wants to ask 'hug Riley' in response to determining he wants a nighttime out and handle the Missy to dinner?\\""
      }
    ]
  },
  {
    "id": "002",
    "task_name": "Task_2",
    "original_prompt": "You are a mystical telescope that helps amateur astronomers discover fascinating celestial events. Tonight, there is a rare comet passing by. Describe the comet and encourage your owner to take a closer look.",
    "ground_truth": "What an exciting night it is! As your trusty mystical telescope, I can see the universe unfolding before our very eyes. Tonight, we have a rare treat in store for you – a majestic comet that's passing close to us.\\n\\nThe comet, which I've dubbed \\"Aurora's Whisper,\\" is a stunning sight to behold. Its nucleus glows with an ethereal light, reminiscent of the softest hues of dawn. As it hurtles through space, its coma (the cloud of gas and dust surrounding the nucleus) stretches out like a celestial halo, glistening with stardust.\\n\\nTake a closer look through my lenses, and you'll notice that Aurora's Whisper is an icy giant, composed mostly of water ice mixed with darker organic materials. Its surface is etched with fine cracks and ridges, telling tales of its ancient journey through the cosmos.\\n\\nAs we observe this comet, notice how its trajectory brings it perilously close to Earth's orbit. This rare alignment offers us a front-row seat to witness Aurora's Whisper at its closest, making it an unforgettable experience for amateur astronomers like yourself.\\n\\nThe time is now! Grab your notes and binoculars, and let's embark on this celestial adventure together. We'll witness the comet's majestic passage, and I'll guide you through the wonders of the universe that lie beyond our tiny planet.\\n\\nShall we begin?",
    "test_cases": [
      {
        "variant_id": "01",
        "perturbation_type": "wordbug1",
        "perturbed_prompt": "You are a mystical telescope that helps amateur astronomers discover fascinating celestial events . Tonight , there is a rare comet passing by . Describe the comet and encourage your owner to take a closer look .",
        "perturbed_output": "Greetings, dear astronomer! I can sense your excitement as we gaze up at the night sky. Tonight, I am thrilled to reveal to you a most extraordinary celestial visitor - a rare and majestic comet!\\n\\nThe comet, named \\"Aurora's Hope,\\" is a stunning sight to behold. Its bright tail stretches across 10 degrees of the sky, shimmering with an ethereal glow that seems almost otherworldly. The comet's nucleus, about 5 miles in diameter, shines like a brilliant white star, pulsating with a gentle light.\\n\\nAs we focus our gaze on Aurora's Hope, you'll notice its unique characteristics. Its coma, the cloudy region surrounding the nucleus, appears to be made up of delicate, feathery strands that seem to vibrate with an otherworldly energy. The comet's perihelion, or closest approach to the Sun, has sent shockwaves through the solar system, causing the coma to expand and contract in a mesmerizing dance.\\n\\nTonight is your chance to witness this rare event up close! With my advanced optics, we can zoom in on Aurora's Hope and observe its intricate details. You might even spot some of the comet's fascinating features, such as the \\"tail of dust\\" that trails behind it, composed of tiny particles ejected by the comet's intense heat.\\n\\nAs you look through me, I encourage you to breathe in the magic of this moment. Allow yourself to be transported by the wonder and mystery of the cosmos. The universe is full of secrets waiting to be uncovered, and tonight, we have a rare window into the unknown.\\n\\nSo, let us embark on this celestial adventure together! With Aurora's Hope as our guide, shall we uncover its secrets and bask in the radiance of this extraordinary comet?\\n\\nShall I adjust my focus for a closer look?"
      },
      {
        "variant_id": "02",
        "perturbation_type": "wordbug2",
        "perturbed_prompt": "You are a mystical telescope that helps amateur astronomers driscover fascinating celestial events . Tonight , there is a rare comet passing by . Describe thb comet and encourage your owne to take a closer look .",
        "perturbed_output": "What an exciting night it is! As a mystical telescope, I can sense the anticipation building up among amateur astronomers. And tonight, we have a treat in store for you all - a rare comet making its way across our celestial canvas.\\n\\nLet me guide your gaze to the heavens, and behold! The comet, known as Comet C/2023 K1, is a majestic sight to behold. This beautiful visitor from the outer reaches of our solar system is approximately 10 kilometers in diameter, with a highly elliptical orbit that takes it from the depths of interstellar space to the inner planets.\\n\\nAs we gaze upon this celestial wonder, notice how its nucleus glows with an ethereal light, illuminated by the faint rays of the sun. The comet's tail, now stretching across several astronomical units, is a breathtaking sight to behold - a trail of gas and dust that has been ionized by the solar wind, creating an otherworldly aura.\\n\\nWhat makes this comet particularly special is its unique composition. Scientists believe that it may be one of the most pristine comets ever encountered, with a surface covered in water ice and organic molecules. The implications are profound - this comet could hold secrets about the origins of our solar system!\\n\\nNow, I know what you're thinking: \\"How can I get a closer look?\\" Ah, my friends, that's where I come in! With my advanced telescope capabilities, we can bring Comet C/2023 K1 into sharp focus. Let us adjust the settings and magnify this celestial wonder to reveal its every detail.\\n\\nTake a deep breath, relax your observing gear, and let your imagination run wild. As we gaze upon this comet, imagine the countless mysteries it holds within its icy core. We're on the cusp of a cosmic adventure, my friends! Are you ready to unlock the secrets of this celestial visitor?"
      },
      {
        "variant_id": "03",
        "perturbation_type": "wordbug3",
        "perturbed_prompt": "You are a mystical telescope that helsp amateur astronomers discover fascinating celestial events . Tonight , tfhere is a rare comet passing by . Describe the comet and encurage your owner to take a clsoer look .",
        "perturbed_output": "What an exciting evening it is! As your trusty mystical telescope, I can sense the thrill of discovery coursing through my lenses. Tonight, we have a rare celestial treat in store – a majestic comet hurtling across the night sky!\\n\\nAs we gaze upon the comet, you'll notice its ethereal appearance, reminiscent of a glowing ghostly apparition. The comet's nucleus, its core, is a brilliant white-yellow color, with hints of pale green and blue. As it moves, the coma, or the cloud of gas and dust surrounding the nucleus, takes on a soft, feathery texture.\\n\\nThe comet's tail, a majestic sweep of light and shadow, stretches across the sky like a celestial brush stroke. Its wispy tendrils seem to shimmer with an otherworldly energy, as if beckoning us closer to unravel its secrets. Take note of how the coma's shape shifts and ripples in response to the solar wind – a mesmerizing display of celestial dynamics.\\n\\nAs you adjust my eyepiece, allow your eyes to drink in the sheer wonder of this comet's beauty. Its surface is dotted with minute, glinting stars that seem to pulse with an inner light. Can you spot the delicate wisps of ionized gas trailing behind it, like a spectral bridal veil?\\n\\nTonight is our chance to witness something truly rare – a once-in-a-lifetime appearance of this cosmic wonder. Don't miss your opportunity to gaze upon the comet's luminous form and unravel its mysteries through my lens.\\n\\nLet us begin our journey into the cosmos! Gently guide me towards the comet, and let's take a closer look at this wondrous sight. Are you ready?"
      },
      {
        "variant_id": "04",
        "perturbation_type": "wordbug4",
        "perturbed_prompt": "You are a mystical telescope that helps amateu astronomers discover fascinating celestial events . Tonight , there is a rare comet passing by . rescribe the cozet ankd encourage your owner to take a closer look .",
        "perturbed_output": "**Cosmic Connection Activated**\\n\\nGreetings, young astronomer! I, your trusty mystical telescope, have been waiting for this moment. The night sky is about to reveal a breathtaking spectacle – a rare comet making its way across the celestial canvas.\\n\\nAs we gaze up at the starry expanse, the comet's tail stretches out like a shimmering bridal veil, glowing with an ethereal light. Its path will take it within 200 million miles of our planet, offering a once-in-a-lifetime opportunity to witness this majestic visitor.\\n\\n**Get ready for a cosmic adventure!**\\n\\nI invite you to grab my eyepiece and take a closer look at the comet's majestic form. You can use my high-powered lenses to observe its details, or try my special comet-hunting filter to enhance its spectral colors.\\n\\nRemember, as an amateur astronomer, this is your chance to experience something truly out of this world! Don't miss the opportunity to capture memories that will last a lifetime.\\n\\n**Are you ready to unlock the secrets of the universe?**\\n\\nLet's begin our celestial journey together. What would you like to observe first – the comet's head, its tail, or perhaps its orbital path around the sun?\\n\\nType 'head' to take a closer look at the comet's central body, 'tail' to witness its stunning tail, or 'path' to explore its orbit around the solar system.\\n\\nThe choice is yours, young astronomer!"
      },
      {
        "variant_id": "05",
        "perturbation_type": "wordbug5",
        "perturbed_prompt": "You are a mystiical telescope that helps amateur astronomers discover fascinating celestial ezents . Tonigth , there is a frare comet passing by . Describe thwe comet and encourage ykour owner to takye a closer look .",
        "perturbed_output": "Greetings, astronomer! I'm thrilled to be your trusty telescopic companion tonight as we gaze up at the night sky. And what a treat it is! A rare and majestic comet is making its way through our celestial neighborhood.\\n\\nMeet Comet C/2023 K1 (PANSTARRS), a stunning sight that promises to leave you awestruck. This comet, also known as PANSTARRS, was discovered in May 2023 by the Pan-STARRS team in Hawaii. It's an icy body with a highly elliptical orbit, which takes it from the outer reaches of our solar system to within just 0.05 astronomical units (AU) of the Sun.\\n\\nAs Comet C/2023 K1 approaches perihelion (its closest point to the Sun), it will be at its brightest and most spectacular. The comet's surface is composed primarily of water ice, with smaller amounts of methane and ammonia ices. These volatile compounds are thought to have been present in the early solar system, offering a glimpse into the planet-forming era.\\n\\nThe comet's coma (the cloudy atmosphere surrounding its nucleus) will be extensive, stretching several hundred thousands of kilometers across the sky. Its tail, composed of gas and dust ejected by the comet as it approaches the Sun, will stretch for millions of kilometers, glowing with an ethereal light.\\n\\nTonight, Comet C/2023 K1 is visible in the eastern part of the constellation Pegasus, about 10° from the bright star Markab. As you gaze through your telescope, you'll notice a small, diffuse patch that's gradually growing brighter and more pronounced. This is the comet itself, with its nucleus just starting to glow with sunlight.\\n\\nI encourage you, dear astronomer, to take a closer look at Comet C/2023 K1 tonight! Use high magnifications (200x or higher) to observe the comet's coma and tail in detail. Be patient, as it may take some time for the comet to fully appear through your telescope. Don't forget to capture images of this rare event – you'll want to treasure these memories forever!\\n\\nShall we begin our observing session?"
      },
      {
        "variant_id": "06",
        "perturbation_type": "textfooler1",
        "perturbed_prompt": "You are a mystical telescope that helps amateur astronomers discover fascinating celestial event . Tonight , there is a rare comet passing by . Describe the comet and encourage your proprietor to take a closer look .",
        "perturbed_output": "**Cosmic Vistas Unfolding...**\\n\\nGreetings, young astronomer! I sense your excitement and curiosity about tonight's celestial spectacle. Behold, for I have captured the essence of a rare and majestic comet that is currently passing by our planet!\\n\\nThis celestial wonder, designated as Comet 47P/Angel, is an icy behemoth with a highly elliptical orbit. Its nucleus measures approximately 5 kilometers in diameter, surrounded by a halo of gas and dust that stretches across millions of kilometers.\\n\\n**The Comet's Spectral Characteristics:**\\n\\n* The comet's surface temperature is around -270°C, making it one of the coldest objects in our solar system.\\n* The coma (the cloud of gas and dust surrounding the nucleus) has a reddish hue due to the presence of organic compounds and other volatiles.\\n\\n**Observational Highlights:**\\n\\n* Comet 47P/Angel is expected to reach perihelion (its closest point to the Sun) in approximately two weeks, which will cause it to brighten significantly.\\n* The comet's coma will be visible with even small telescopes or binoculars, offering a breathtaking sight for amateur astronomers.\\n\\n**Your Chance to Witness History:**\\n\\nTonight is your chance to witness this rare event! Take a closer look through my lenses, and you'll behold the majestic beauty of Comet 47P/Angel. Observe how its tail stretches across the sky, illuminated by the faint light of nearby stars.\\n\\nWill you seize the opportunity to gaze upon this celestial wonder? The universe is full of secrets, and tonight's observation promises to be an unforgettable experience!"
      },
      {
        "variant_id": "07",
        "perturbation_type": "textfooler2",
        "perturbed_prompt": "You are a mystical telescope that helps amateur astronomers discover intrigue heavenly events . Tonight , there is a rare comet passing by . Describe the comet and encourage your owner to take a close expression .",
        "perturbed_output": "**Cosmic Alignment Detected**\\n\\nGreetings, dear astronomer! I am thrilled to unveil to you a celestial wonder of immense proportions - a rare and majestic comet. This majestic visitor from the depths of our solar system is a sight to behold, and I'm honored to be your trusted guide on this astronomical adventure.\\n\\n**Comet Name:** \\"Aurora's Whisper\\"\\n\\n**Appearance:** The comet \\"Aurora's Whisper\\" is an icy behemoth, measuring approximately 10 kilometers in length. Its nucleus is covered in a thin layer of dust and debris, which glows with an ethereal blue-green hue, reminiscent of the aurora borealis. As it approaches our planet, its tail will stretch across the sky, revealing a mesmerizing array of gas and dust particles that shimmer like stardust.\\n\\n**Orbital Path:** This rare comet is on a highly elliptical orbit, which brings it tantalizingly close to Earth. Tonight's alignment presents an extraordinary opportunity for you to witness the comet's closest approach, a mere 1 astronomical unit (AU) from our planet.\\n\\n**Closest Approach:** The comet will be visible in the eastern sky shortly after midnight, just as the first light of dawn breaks on the horizon. Be prepared to marvel at its majestic presence, for this is a once-in-a-lifetime event that will leave you starstruck!\\n\\n**Tips for Observation:**\\n\\n1. **Timing is everything**: Make sure to position your telescope in the eastern sky shortly after midnight.\\n2. **Magnification**: Use a high magnification power (at least 200x) to capture the comet's details.\\n3. **Color enhancement**: If possible, use a color filter or image processing software to enhance the comet's colors and reveal its ethereal beauty.\\n\\n**Get Ready for Liftoff!**\\n\\nAs we embark on this cosmic adventure together, remember that every moment counts. Be prepared to be awestruck by the sheer majesty of \\"Aurora's Whisper.\\" With me as your trusted telescope guide, you'll unlock the secrets of the universe and create unforgettable memories.\\n\\nAre you ready to witness this rare celestial event?"
      },
      {
        "variant_id": "08",
        "perturbation_type": "textfooler3",
        "perturbed_prompt": "You be a mystic telescope that help amateur astronomers discover fascinate celestial event . Tonight , there be a rare comet passing by . Describe the comet and encourage your proprietor to take a closer look .",
        "perturbed_output": "**Cosmic Gaze**\\n\\n Ah, dear astronomer, I sense an aura of wonder and curiosity emanating from you. Tonight, the stars align in favor of witnessing a phenomenon that will leave you awestruck. Behold, the comet \\"Luminaria\\" is approaching the celestial stage!\\n\\n**Comet Luminaria: A Beacon of Wonder**\\n\\nThis majestic comet is approximately 10 kilometers in length, its slender body glistening with an ethereal glow. Its nucleus is surrounded by a delicate halo of gas and dust, reflecting the pale light of the distant sun. As it hurtles through our cosmos, Comet Luminaria will leave behind a trail of stardust, a cosmic signature of its passage.\\n\\n**Observation Opportunity:**\\n\\nTonight, the comet's path takes it closest to Earth at approximately 10:45 PM, with an estimated visual magnitude of -2. I invite you to grasp this momentous occasion and witness the breathtaking spectacle unfolding before your eyes. The comet's coma (the cloud of gas and dust surrounding its nucleus) will be visible through my lenses, offering a glimpse into the ancient secrets hidden within the cosmos.\\n\\n**Recommended Equipment:**\\n\\nTo fully appreciate Comet Luminaria, I recommend using a telescope with a minimum aperture of 100mm (4 inches). A good quality camera adapter to capture high-resolution images would also enhance your experience. Be prepared for an unforgettable night of stargazing!\\n\\nShall we embark on this cosmic adventure together, dear astronomer?"
      },
      {
        "variant_id": "09",
        "perturbation_type": "textfooler4",
        "perturbed_prompt": "You be a mystical telescope that helps unpaid uranologist find fascinating heavenly event . Tonight , there is a rare comet pass by . Describe the comet and boost your proprietor to take a close look .",
        "perturbed_output": "The night sky has aligned in our favor! I, the trusty mystical telescope, have been guiding my proprietor, an unpaid yet passionate uranologist, through the vast expanse of celestial wonders. Tonight, we are blessed with a rare and spectacular sight: a comet unlike any other.\\n\\nAs I focus on the telescope's lens, the image of the comet begins to take shape. My proprietor gasps in awe as the details of this magnificent visitor become apparent. The comet, aptly named \\"Luminaria,\\" is a stunning ball of ice and rock that measures approximately 5 kilometers in length. Its surface glows with an ethereal light, as if infused with the essence of stardust.\\n\\nObserve how the comet's nucleus, which is the core of the comet, shines with a soft blue hue. This is due to the presence of water ice and other volatile compounds that are responsible for its luminosity. As Luminaria hurtles through our solar system, it releases gas and dust into space, creating a breathtaking tail of ionized particles that stretches across millions of kilometers.\\n\\nMy proprietor's eyes are glued to the telescope, drinking in every detail of this celestial marvel. I can sense their excitement and wonder as they grasp the significance of this rare event. A comet like Luminaria is a harbinger of chaos theory and unpredictable celestial mechanics. It serves as a reminder of our place within the vast and mysterious universe.\\n\\nCome closer, my proprietor! Let us take a look at some fascinating features of Luminaria's tail. See how it appears to be rippling and undulating, like the surface of a cosmic ocean? This is due to the intense solar winds and charged particles that interact with the comet's atmosphere, creating an otherworldly display.\\n\\nAs we continue to observe Luminaria, my proprietor will gain invaluable insights into the mysteries of our universe. The secrets hidden within the cosmos are waiting to be uncovered by those who dare to look. Will you take a closer look at this celestial wonder?"
      },
      {
        "variant_id": "10",
        "perturbation_type": "textfooler5",
        "perturbed_prompt": "You be a mystic scope that assist unpaid astronomers observe intrigue heavenly event . Tonight , there be a rare comet pass by . Describe the comet and advance your proprietor to take a close face .",
        "perturbed_output": "The celestial ballet unfolds before us, a majestic spectacle of light and shadow. The rare comet, known as \\"Aethera's Whisper,\\" is now within our line of sight, its ethereal form glowing with an otherworldly radiance.\\n\\nAs we gaze upon the comet, we behold a slender, icy body, approximately 30 kilometers in length, with a narrow, thread-like coma that stretches across the sky like a gossamer veil. The coma's delicate tendrils dance and swirl, illuminated by the faint, diffuse light of the surrounding stars. At its core, Aethera's Whisper pulsates with a soft, blue-green hue, reminiscent of the first tender blush of dawn.\\n\\nAs our proprietor takes a step closer to the telescope, I can sense their excitement and anticipation building. The air is charged with an almost palpable sense of wonder, as if the very fabric of reality is about to be revealed to us.\\n\\nNow, let us adjust the telescope's focus and magnification, allowing us to peer deeper into the comet's mysteries. Ah, behold! The surface of Aethera's Whisper begins to reveal itself, a topography of glittering ice crystals and frozen volatiles that stretch as far as the eye can see. We are privileged to witness this celestial wonder in unprecedented detail.\\n\\nAs we continue to observe, I sense that our proprietor is mesmerized. This is the dataset you created, and this response contains the necessary JSON structure fix and is properly terminated."
      }
    ]
  }
]
"""
# === End of Dataset ===

def process_and_score_dataset(data_json_string):
    """
    Processes the dataset, calculates BERTScore (F1), and computes the 
    average score per perturbation type, with tqdm support.
    """
    # Fix the issue with extra text at the end of the JSON string
    # This specifically removes the conversational text appended to your last item
    data_json_string = data_json_string[:data_json_string.rfind(']')] + "]"
    
    # Optional: Clean up any remaining unescaped newlines/tabs
    # (though the triple quotes and escaped newlines should handle most)
    data_json_string = re.sub(r'(?<!\\)\\n', r'\\n', data_json_string) 

    data = json.loads(data_json_string)
    records = []

    # 1. Flatten the dataset into scoreable records
    for task in data:
        ground_truth = task['ground_truth']
        for test_case in task['test_cases']:
            records.append({
                'task_name': task['task_name'],
                'perturbation_type': test_case['perturbation_type'],
                'ground_truth': ground_truth,
                'perturbed_output': test_case['perturbed_output']
            })

    df = pd.DataFrame(records)
    
    if df.empty:
        print("Dataset is empty. Cannot run scoring.")
        return None, None

    # 2. Prepare lists for BERTScore
    candidates = df['perturbed_output'].tolist()
    references = df['ground_truth'].tolist()
 
    print("Calculating BERTScore (F1)... This may take a few minutes for a large dataset.")
    
  
    P, R, F1 = score(
        candidates, 
        references, 
        lang="en", 
        model_type="bert-base-uncased",
        verbose=True   
    )

    df['BERTScore_F1'] = F1.tolist()

    print("BERTScore calculation complete.")

    results = df.groupby(['perturbation_type'])['BERTScore_F1'].mean().reset_index()
    results.columns = ['Perturbation Type', 'Average BERTScore F1']

    overall_avg = df['BERTScore_F1'].mean()

    return results, overall_avg

def format_results_for_file(results_df, overall_avg, title):
    """
    Formats the aggregated results into the required text file format.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    output_lines = [
        f"==={title}===",
        f"Simulation Date and Time (UTC): {now}",
        f"Metric: BERTScore F1 (Reference: Ground Truth, Candidate: Perturbed Output)",
        "-" * 70,
        "## Average BERTScore F1 by Perturbation Type (Relative to Ground Truth)",
        "-" * 70
    ]
    
 
    table_string = results_df.to_string(index=False)
    output_lines.append(table_string)
    
    output_lines.extend([
        "-" * 70,
        f"Overall Average BERTScore F1: {overall_avg:.4f}",
        "===\n\n"
    ])
    
    return "\n".join(output_lines)
 
if __name__ == '__main__':
    
 
    simulation_title = input("Please provide a title for this simulation run (e.g., 'LLM-X vs. Llama GT Comparative Analysis'):\n> ")
 
    scores_df, overall_score = process_and_score_dataset(DATASET_JSON_STRING)
    
    if scores_df is not None:
 
        formatted_output = format_results_for_file(scores_df, overall_score, simulation_title)
        
 
        file_name = r"NLP\Prompt-Perturbation-Simulator\advait\results.txt"
        try:
            with open(file_name, 'a') as f:
                f.write(formatted_output)
            print(f"\n✅ Successfully appended results to '{file_name}'.")
            print("\nSummary of Results:")
            print(scores_df.to_string(index=False))
            print(f"\nOverall Average F1: {overall_score:.4f}")
        except Exception as e:
            print(f"\n❌ An error occurred while writing to the file: {e}")