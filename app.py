from flask import Flask, request, jsonify, render_template
from utils.medicine import search_medicine
from models.llm import create_llm_chain

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('Contact.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    print(query)
    
    if not query:
        return jsonify({"error": "No query provided"}), 400

    # Create LLM chain
    llm_chain = create_llm_chain()
    
    try:
        # Search for medicines
        medicines = search_medicine(query)
        
        if medicines:
            # Extract and format medicine information
            primary_medicine = medicines.get('primary', None)
            alternative_medicines = medicines.get('alternatives', [])
            
            # Prepare formatted information
            if primary_medicine:
                primary_info = (
                    f"Name: {primary_medicine['Name']}\n"
                    f"Uses: {primary_medicine['Uses']}\n"
                    f"Composition: {primary_medicine['Composition']}\n"
                    f"Manufacturer: {primary_medicine['Manufacturer']}\n"
                )
            else:
                primary_info = "No primary medicine found."
            
            if alternative_medicines:
                alternatives_info = "\n".join(
                    [
                        f"- {alt['Name']} ({alt['Composition']}), Uses: {alt['Uses']}\n"
                        for alt in alternative_medicines
                    ]
                )
            else:
                alternatives_info = "No alternative medicines found."
            
            medicine_info = f"Primary Medicine:\n{primary_info}\n\nAlternatives:\n{alternatives_info}"
        else:
            medicine_info = f"Sorry, we couldn’t identify any medicines related to {query}. Let us know how else we can assist you.."

    except Exception as e:
        # Log the error and provide a fallback response
        print(f"Error during medicine identification: {e}")
        medicines = None
        medicine_info = "An error occurred while identifying medicines. Please try again later."

    # Always run LLM chain
    response = llm_chain.run(question=query)
    print(f"LLM Response: {response}")

    return jsonify({
        "medicines": medicines if medicines else {"message": "No medicines found or an error occurred"},
        "formatted_info": medicine_info,
        "llm_response": response
    })

if __name__ == '__main__':
    app.run(debug=True)
