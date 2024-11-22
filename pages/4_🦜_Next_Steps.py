# Import necessary libraries
import streamlit as st

st.set_page_config(
    page_title = "Next Steps",
    page_icon = "🦜")

# Title and description
st.markdown(
    """
    <h2 style = "text-align: center; color: #69503c;">What Now?</h2>
    """,
    unsafe_allow_html = True,
)

# Create a centered layout with st.columns
col1, col2, col3 = st.columns([1, 2, 1])

with col2:  # Center column
    st.image('happy_bird.gif', use_column_width=True)

# Sidebar information
with st.sidebar:
    st.header("🦜 Next Steps")
    st.write("""
    On this page, you can:
    1. View tips on what to do if your pet will be adopted
    2. View tips on how you can increase the chance your pet will be adopted
    3. Explore additional pet adoption resources
    """)

# If Adopted Section
st.markdown(
    """
    <h3 style="color: green;">If the Pet Will Be Adopted:</h3>
    <p style="font-size: 16px;">
    Congratulations! Your pet has a new home! Here are some next steps to ensure a smooth transition:
    </p>
    <ul style="font-size: 16px;">
        <li>Prepare your home with the necessary supplies to welcome the new pet.</li>
        <li>Communicate with the adopter about care instructions and the pet's history.</li>
        <li>Check in periodically to ensure your pet is adjusting well.</li>
        <li>Join local pet communities for ongoing support and updates.</li>
    </ul>
    <p style="font-size: 16px;">
    It's an exciting time for both you and your pet! Best of luck in the next chapter of your pet’s journey.
    </p>
    """,
    unsafe_allow_html=True,
)

# If Not Adopted Section
st.markdown(
    """
    <h3 style="color: green;">If the Pet Will Not Be Adopted:</h3>
    <p style="font-size: 16px;">
    Don't give up! Even if the prediction isn't in your favor, there are still ways to increase your pet's chances of adoption:
    </p>
    <ul style="font-size: 16px;">
        <li>Improve the pet's profile with updated photos and a detailed description.</li>
        <li>Promote your pet on social media and other pet adoption websites.</li>
        <li>Reach out to other local shelters and rescue groups for additional support.</li>
        <li>Consider fostering the pet to give them more exposure to potential adopters.</li>
        <li>Ensure its health and vaccinations are up to date.</li>
    </ul>
    <p style="font-size: 16px;">
    Remember, persistence is key. Many loving homes are waiting for the right pet.
    </p>
    """,
    unsafe_allow_html=True,
)

# Additional Resources Section
st.markdown(
    """
    <h3 style="color: green;">Additional Resources:</h3>
    <p style="font-size: 16px;">
    Here are some helpful resources for both adopters and pet owners:
    </p>
    <ul style="font-size: 16px;">
        <li><a href="https://www.petfinder.com/" target="_blank">Petfinder</a> – Find adoptable pets across the country.</li>
        <li><a href="https://www.adoptapet.com/" target="_blank">Adopt-a-Pet</a> – A leading pet adoption website.</li>
        <li><a href="https://www.humanesociety.org/" target="_blank">The Humane Society</a> – For pet care, adoption, and advocacy.</li>
    </ul>
    <p style="font-size: 16px;">
    Whether you're looking to adopt or help a pet find their forever home, these resources can guide you.
    </p>
    """,
    unsafe_allow_html=True,
)

# Adoption Tips Section
st.markdown(
    """
    <h3 style="color: green;">Adoption Tips:</h3>
    <p style="font-size: 16px;">
    Here are some general tips for both pet adopters and owners:
    </p>
    <ul style="font-size: 16px;">
        <li>Provide a safe and welcoming environment for your pet.</li>
        <li>Establish a routine and be patient as your pet adjusts to their new home.</li>
        <li>Stay informed about your pet's needs and seek advice if needed.</li>
        <li>Consider fostering a pet if you're unsure about full adoption.</li>
    </ul>
    """,
    unsafe_allow_html=True,
)

# FAQs Section
st.markdown(
    """
    <h3 style="color: green;">Frequently Asked Questions:</h3>
    <p style="font-size: 16px;">
    Here are some common questions and answers about the adoption process:
    </p>
    <ul style="font-size: 16px;">
        <li><strong>How can I improve my pet's chances of adoption?</strong> – Update your pet's profile with great photos, detailed descriptions, and share it on social media.</li>
        <li><strong>What if my pet isn't adopted right away?</strong> – Be patient, improve your pet's profile, and try other adoption platforms and shelters.</li>
        <li><strong>How can I stay in touch with the adopter?</strong> – If possible, communicate with the adopter to ensure a smooth transition for your pet.</li>
    </ul>
    """,
    unsafe_allow_html=True,
)

# Call to Action Section
st.markdown(
    """
    <h3 style="color: green;">Share and Feedback:</h3>
    <p style="font-size: 16px;">
    If you found this app helpful, share it with others who may need guidance in the pet adoption journey. We’d love to hear your feedback!
    </p>
    <p style="font-size: 16px;">
    Together, we can help pets find their forever homes.
    </p>
    """,
    unsafe_allow_html=True,
)