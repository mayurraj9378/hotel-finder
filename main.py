import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import urllib.parse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load sample data (in a real app, you'd connect to a database)
@st.cache_data
def load_sample_data():
    # Sample hotel data
    hotels = pd.DataFrame({
        'hotel_id': range(1, 21),
        'name': [
            'Grand Plaza Hotel', 'Seaside Resort', 'Mountain View Lodge', 'City Center Inn',
            'Luxury Palace Hotel', 'Business Executive Suites', 'Family Fun Resort', 'Historic Boutique Hotel',
            'Modern Minimalist Hotel', 'Beachfront Paradise', 'Urban Oasis Hotel', 'Countryside Retreat',
            'Skyline Hotel', 'Heritage Grand Hotel', 'Tech-Savvy Suites', 'Wellness Spa Resort',
            'Adventure Base Camp', 'Romantic Getaway Inn', 'Budget Friendly Motel', 'Exclusive Club Resort'
        ],
        'location': ['Downtown', 'Beach', 'Mountains', 'City Center', 'Uptown', 'Business District', 
                    'Suburban', 'Old Town', 'Arts District', 'Beachfront', 'Urban', 'Countryside', 
                    'City View', 'Historic District', 'Tech Hub', 'Lakeside', 'National Park', 
                    'Countryside', 'Highway Access', 'Private Island'],
        'price_category': ['Luxury', 'Premium', 'Mid-range', 'Budget', 'Luxury', 'Business', 'Family', 
                           'Boutique', 'Modern', 'Premium', 'Mid-range', 'Budget', 'Business', 'Luxury', 
                           'Modern', 'Premium', 'Adventure', 'Romantic', 'Budget', 'Luxury'],
        'avg_rating': [4.7, 4.5, 4.2, 3.8, 4.9, 4.3, 4.1, 4.4, 4.0, 4.6, 4.2, 3.9, 4.1, 4.8, 4.0, 4.5, 4.3, 4.7, 3.5, 4.9],
        'amenities': [
            'Pool, Spa, Restaurant, Gym, WiFi', 'Beach access, Pool, Restaurant, WiFi',
            'Hiking trails, Restaurant, Fireplace, WiFi', 'Restaurant, WiFi, Business Center',
            'Pool, Spa, Multiple Restaurants, Gym, WiFi, Concierge', 'Business Center, WiFi, Restaurant, Gym',
            'Kids Club, Pool, Restaurant, Playground, WiFi', 'Room Service, Restaurant, WiFi, Historic Tours',
            'WiFi, Restaurant, Modern Art, Gym', 'Private Beach, Pool, Restaurant, Water Sports, WiFi',
            'Restaurant, WiFi, Rooftop Bar, Gym', 'Garden, WiFi, Restaurant, Nature Trails',
            'Restaurant, WiFi, Business Center, City Tours', 'Pool, Spa, Fine Dining, WiFi, Heritage Tours',
            'High-speed WiFi, Smart Rooms, Co-working Space, Restaurant', 'Spa, Yoga, Pool, Healthy Restaurant, WiFi',
            'Guided Tours, Equipment Rental, Restaurant, WiFi', 'Spa, Fine Dining, WiFi, Couples Activities',
            'WiFi, Basic Breakfast, Parking', 'Private Beach, Pool, Spa, Gourmet Dining, Water Sports, WiFi'
        ]
    })
    
    # Sample customer data
    np.random.seed(42)
    n_customers = 500
    customer_ids = range(1, n_customers + 1)
    age = np.random.randint(18, 75, n_customers)
    gender = np.random.choice(['M', 'F'], n_customers)
    income_levels = np.random.choice(['Low', 'Medium', 'High'], n_customers, p=[0.3, 0.5, 0.2])
    travel_purpose = np.random.choice(['Business', 'Leisure', 'Family', 'Romantic'], n_customers)
    preferred_amenities = np.random.choice(['WiFi', 'Pool', 'Spa', 'Restaurant', 'Gym', 'Beach access'], n_customers)
    
    customers = pd.DataFrame({
        'customer_id': customer_ids,
        'age': age,
        'gender': gender,
        'income_level': income_levels,
        'travel_purpose': travel_purpose,
        'preferred_amenity': preferred_amenities
    })
    
    # Sample booking data
    n_bookings = 1000
    booking_ids = range(1, n_bookings + 1)
    customer_ids_bookings = np.random.choice(customer_ids, n_bookings)
    hotel_ids_bookings = np.random.choice(hotels['hotel_id'], n_bookings)
    booking_date = pd.date_range(start='2022-01-01', end='2023-06-30', periods=n_bookings)
    length_of_stay = np.random.randint(1, 10, n_bookings)
    booking_value = np.random.uniform(100, 1000, n_bookings)
    canceled = np.random.choice([0, 1], n_bookings, p=[0.8, 0.2])
    lead_time = np.random.randint(1, 100, n_bookings)
    season = np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n_bookings)
    booking_channel = np.random.choice(['Direct', 'OTA', 'Travel Agent', 'Corporate'], n_bookings)
    
    bookings = pd.DataFrame({
        'booking_id': booking_ids,
        'customer_id': customer_ids_bookings,
        'hotel_id': hotel_ids_bookings,
        'booking_date': booking_date,
        'length_of_stay': length_of_stay,
        'booking_value': booking_value,
        'canceled': canceled,
        'lead_time': lead_time,
        'season': season,
        'booking_channel': booking_channel
    })
    
    # Sample review data
    n_reviews = 800
    review_ids = range(1, n_reviews + 1)
    customer_ids_reviews = np.random.choice(customer_ids, n_reviews)
    hotel_ids_reviews = np.random.choice(hotels['hotel_id'], n_reviews)
    
    review_texts = [
        "Great hotel, really enjoyed my stay!",
        "Good value for money, but the room was a bit small.",
        "Excellent service and friendly staff.",
        "Beautiful location, but the food could be better.",
        "Perfect for a business trip, good amenities.",
        "Loved the pool and spa facilities!",
        "Disappointing experience, the room was not clean.",
        "Amazing views and comfortable beds.",
        "The staff was very helpful and attentive.",
        "Noisy location, couldn't sleep well."
    ]
    
    review_text = np.random.choice(review_texts, n_reviews)
    rating = np.random.randint(1, 6, n_reviews)
    review_date = pd.date_range(start='2022-01-01', end='2023-06-30', periods=n_reviews)
    
    reviews = pd.DataFrame({
        'review_id': review_ids,
        'customer_id': customer_ids_reviews,
        'hotel_id': hotel_ids_reviews,
        'review_text': review_text,
        'rating': rating,
        'review_date': review_date
    })
    
    return hotels, customers, bookings, reviews

# Function to get hotels from OpenStreetMap API
def get_hotels_osm(location, keyword=None):
    # URL encode the location
    encoded_location = urllib.parse.quote(location)
    
    # Use Nominatim for geocoding (finding coordinates from location name)
    geocode_url = f"https://nominatim.openstreetmap.org/search?q={encoded_location}&format=json"
    
    headers = {
        "User-Agent": "HotelFinderApp/1.0"  # Nominatim requires a user agent
    }
    
    try:
        # Get coordinates for the location
        response = requests.get(geocode_url, headers=headers)
        response.raise_for_status()
        location_data = response.json()
        
        if not location_data:
            st.sidebar.error("Location not found in OpenStreetMap")
            return []
        
        # Extract latitude and longitude
        lat = location_data[0]["lat"]
        lon = location_data[0]["lon"]
        
        # Use Overpass API to find hotels
        overpass_url = "https://overpass-api.de/api/interpreter"
        
        # Build query based on keyword
        amenity_filter = ""
        if keyword:
            # Add keyword to search in the name tag
            amenity_filter = f'["name"~"{keyword}",i]'
        
        # Query for hotels within 5km radius
        overpass_query = f"""
        [out:json];
        (
          node["tourism"="hotel"]{amenity_filter}(around:5000,{lat},{lon});
          way["tourism"="hotel"]{amenity_filter}(around:5000,{lat},{lon});
          relation["tourism"="hotel"]{amenity_filter}(around:5000,{lat},{lon});
          node["building"="hotel"]{amenity_filter}(around:5000,{lat},{lon});
          way["building"="hotel"]{amenity_filter}(around:5000,{lat},{lon});
          relation["building"="hotel"]{amenity_filter}(around:5000,{lat},{lon});
        );
        out body;
        """
        
        # Respect rate limits
        time.sleep(1)
        
        # Send query
        response = requests.post(overpass_url, data=overpass_query)
        response.raise_for_status()
        hotels_data = response.json()
        
        hotels = []
        for element in hotels_data.get("elements", []):
            tags = element.get("tags", {})
            
            # Format address
            address_parts = []
            if tags.get("addr:housenumber"):
                address_parts.append(tags.get("addr:housenumber"))
            if tags.get("addr:street"):
                address_parts.append(tags.get("addr:street"))
            if tags.get("addr:city"):
                address_parts.append(tags.get("addr:city"))
            if tags.get("addr:postcode"):
                address_parts.append(tags.get("addr:postcode"))
            
            address = ", ".join(address_parts) if address_parts else "Address not available"
            
            # Determine latitude and longitude
            if element["type"] == "node":
                hotel_lat = element.get("lat")
                hotel_lon = element.get("lon")
            else:
                # For ways and relations, use the original search coordinates
                hotel_lat = lat
                hotel_lon = lon
            
            # Get hotel details
            hotel = {
                "Name": tags.get("name", "Unnamed Hotel"),
                "Address": address,
                "Stars": tags.get("stars", "N/A"),
                "Phone": tags.get("phone", "N/A"),
                "Website": tags.get("website", "N/A"),
                "lat": hotel_lat,
                "lon": hotel_lon
            }
            
            hotels.append(hotel)
        
        return hotels
    
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Request Error: {str(e)}")
        return []

# Customer Segmentation using K-means clustering
def cluster_customers(customers):
    # Prepare the data
    # Convert categorical variables to numeric using one-hot encoding
    customers_encoded = pd.get_dummies(
        customers, 
        columns=['gender', 'income_level', 'travel_purpose', 'preferred_amenity']
    )
    
    # Scale the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customers_encoded.drop('customer_id', axis=1))
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    customers_encoded['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Merge clusters back to original dataframe
    customers_with_clusters = customers.copy()
    customers_with_clusters['cluster'] = customers_encoded['cluster']
    
    # Get cluster insights
    cluster_insights = {}
    for cluster in customers_with_clusters['cluster'].unique():
        cluster_data = customers_with_clusters[customers_with_clusters['cluster'] == cluster]
        
        cluster_insights[cluster] = {
            'size': len(cluster_data),
            'avg_age': cluster_data['age'].mean(),
            'top_purpose': cluster_data['travel_purpose'].mode()[0],
            'top_amenity': cluster_data['preferred_amenity'].mode()[0],
            'income_distribution': cluster_data['income_level'].value_counts().to_dict()
        }
    
    return customers_with_clusters, cluster_insights

# Booking Cancellation Prediction
def predict_cancellations(bookings, customers):
    # Merge bookings with customer data to get more features
    data = bookings.merge(customers, on='customer_id')
    
    # Prepare the data for modeling
    features = ['length_of_stay', 'booking_value', 'lead_time', 'age']
    
    # Add categorical features
    categorical_features = ['season', 'booking_channel', 'income_level', 'travel_purpose']
    for feature in categorical_features:
        dummies = pd.get_dummies(data[feature], prefix=feature)
        features.extend(dummies.columns)
        data = pd.concat([data, dummies], axis=1)
    
    X = data[features]
    y = data['canceled']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return rf_model, accuracy, cm, feature_importance

# Review Sentiment Analysis
def analyze_reviews(reviews):
    # Calculate sentiment scores for each review
    reviews['sentiment_scores'] = reviews['review_text'].apply(
        lambda text: sid.polarity_scores(text)
    )
    
    # Extract compound sentiment score
    reviews['sentiment'] = reviews['sentiment_scores'].apply(
        lambda score_dict: score_dict['compound']
    )
    
    # Categorize sentiment
    reviews['sentiment_category'] = reviews['sentiment'].apply(
        lambda score: 'Positive' if score > 0.2 else ('Negative' if score < -0.2 else 'Neutral')
    )
    
    # Analyze reviews by hotel
    hotel_sentiment = reviews.groupby('hotel_id').agg(
        avg_rating=('rating', 'mean'),
        avg_sentiment=('sentiment', 'mean'),
        positive_reviews=('sentiment_category', lambda x: sum(x == 'Positive')),
        neutral_reviews=('sentiment_category', lambda x: sum(x == 'Neutral')),
        negative_reviews=('sentiment_category', lambda x: sum(x == 'Negative')),
        review_count=('review_id', 'count')
    ).reset_index()
    
    return reviews, hotel_sentiment

# Hotel Recommendation System
def recommend_hotels(hotels, customer_profile, reviews):
    # Extract customer preferences
    preferred_amenities = customer_profile['preferred_amenities']
    purpose = customer_profile['purpose']
    budget = customer_profile['budget']
    
    # Calculate hotel scores based on amenities match and ratings
    hotel_scores = []
    
    for _, hotel in hotels.iterrows():
        # Calculate amenity match score
        amenities_list = hotel['amenities'].lower().split(', ')
        amenity_match = sum(1 for amenity in preferred_amenities if amenity.lower() in ' '.join(amenities_list))
        amenity_score = amenity_match / len(preferred_amenities) if preferred_amenities else 0
        
        # Get average rating
        rating_score = hotel['avg_rating'] / 5  # Normalize to 0-1 scale
        
        # Calculate purpose match
        if purpose.lower() in hotel['price_category'].lower():
            purpose_score = 1.0
        elif (purpose == 'Business' and hotel['price_category'] in ['Business', 'Modern']):
            purpose_score = 0.8
        elif (purpose == 'Family' and hotel['price_category'] in ['Family', 'Mid-range']):
            purpose_score = 0.8
        elif (purpose == 'Luxury' and hotel['price_category'] in ['Luxury', 'Premium']):
            purpose_score = 0.8
        elif (purpose == 'Budget' and hotel['price_category'] in ['Budget', 'Mid-range']):
            purpose_score = 0.8
        else:
            purpose_score = 0.3
            
        # Calculate budget match (assuming budget categories align with price_categories)
        if budget.lower() == hotel['price_category'].lower():
            budget_score = 1.0
        elif (budget == 'Budget' and hotel['price_category'] == 'Mid-range') or \
             (budget == 'Mid-range' and hotel['price_category'] in ['Budget', 'Premium']):
            budget_score = 0.5
        else:
            budget_score = 0.0
            
        # Combined score with weighted factors
        total_score = (
            0.3 * amenity_score + 
            0.3 * rating_score + 
            0.2 * purpose_score + 
            0.2 * budget_score
        )
        
        hotel_scores.append({
            'hotel_id': hotel['hotel_id'],
            'name': hotel['name'],
            'location': hotel['location'],
            'price_category': hotel['price_category'],
            'avg_rating': hotel['avg_rating'],
            'amenities': hotel['amenities'],
            'amenity_score': amenity_score,
            'rating_score': rating_score,
            'purpose_score': purpose_score,
            'budget_score': budget_score,
            'total_score': total_score
        })
    
    # Sort hotels by score
    recommended_hotels = sorted(hotel_scores, key=lambda x: x['total_score'], reverse=True)
    
    return recommended_hotels

# Content-based filtering for reviews
def content_based_filtering(hotels, reviews):
    # Combine hotel names with their reviews for content analysis
    hotel_reviews = {}
    
    for _, hotel in hotels.iterrows():
        hotel_id = hotel['hotel_id']
        hotel_reviews[hotel_id] = {
            'name': hotel['name'],
            'content': hotel['name'] + " " + hotel['location'] + " " + hotel['price_category'] + " " + hotel['amenities']
        }
        
        # Add reviews content
        hotel_reviews_texts = reviews[reviews['hotel_id'] == hotel_id]['review_text'].tolist()
        if hotel_reviews_texts:
            hotel_reviews[hotel_id]['content'] += " " + " ".join(hotel_reviews_texts)
    
    # Create a dataframe
    hotel_content_df = pd.DataFrame([
        {'hotel_id': hotel_id, 'name': data['name'], 'content': data['content']}
        for hotel_id, data in hotel_reviews.items()
    ])
    
    # Create TF-IDF vectors
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(hotel_content_df['content'])
    
    # Calculate similarity between hotels
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Create a mapping of hotel names to indices
    indices = pd.Series(hotel_content_df.index, index=hotel_content_df['name']).drop_duplicates()
    
    return indices, cosine_sim, hotel_content_df

# Function to get hotel recommendations based on similarity
def get_content_recommendations(hotel_name, indices, cosine_sim, hotel_content_df):
    # Get the index of the hotel
    idx = indices[hotel_name]
    
    # Get similarity scores with all hotels
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort hotels based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top 5 similar hotels (excluding itself)
    sim_scores = sim_scores[1:6]
    
    # Get hotel indices
    hotel_indices = [i[0] for i in sim_scores]
    
    # Return recommended hotels
    recommendations = hotel_content_df.iloc[hotel_indices][['name', 'hotel_id']]
    
    # Add similarity scores
    recommendations['similarity_score'] = [i[1] for i in sim_scores]
    
    return recommendations

# Main application
def main():
    st.set_page_config(page_title="Hotel Intelligence System", page_icon="🏨", layout="wide")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    pages = [
        "Home", 
        "Hotel Search", 
        "Hotel Recommendations", 
        "Customer Segmentation", 
        "Cancellation Prediction", 
        "Review Analysis"
    ]
    selection = st.sidebar.radio("Go to", pages)
    
    # Load data
    hotels, customers, bookings, reviews = load_sample_data()
    
    # Create content-based filtering model
    indices, cosine_sim, hotel_content_df = content_based_filtering(hotels, reviews)
    
    # Analyze reviews
    reviews_analyzed, hotel_sentiment = analyze_reviews(reviews)
    
    # Home page
    if selection == "Home":
        st.title("🏨 Hotel Intelligence System")
        st.write("Welcome to the Hotel Intelligence System powered by Machine Learning!")
        
        st.markdown("""
        ### Features:
        - **Hotel Search**: Find hotels in any location using OpenStreetMap data
        - **Hotel Recommendations**: Get personalized hotel recommendations based on your preferences
        - **Customer Segmentation**: Understand customer segments to provide targeted services
        - **Cancellation Prediction**: Predict booking cancellations to optimize resource planning
        - **Review Analysis**: Analyze customer feedback to improve services
        
        ### How to use:
        1. Use the navigation panel on the left to switch between features
        2. Explore the data visualizations and insights
        3. Try personalized recommendations and search capabilities
        
        ### Sample Dataset:
        This application is using a sample dataset for demonstration purposes.
        """)
        
        # Display sample data tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sample Hotels")
            st.dataframe(hotels[['name', 'location', 'price_category', 'avg_rating']].head())
            
        with col2:
            st.subheader("Bookings Overview")
            bookings_summary = bookings.groupby('canceled').agg(
                count=('booking_id', 'count'),
                avg_value=('booking_value', 'mean'),
                avg_stay=('length_of_stay', 'mean')
            ).reset_index()
            bookings_summary['canceled'] = bookings_summary['canceled'].map({0: 'Confirmed', 1: 'Canceled'})
            st.dataframe(bookings_summary)
            
        # Show some visualizations
        st.subheader("Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            sns.countplot(x='price_category', data=hotels, ax=ax)
            plt.title('Hotels by Price Category')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots()
            sns.histplot(data=reviews, x='rating', kde=True, ax=ax)
            plt.title('Distribution of Hotel Ratings')
            st.pyplot(fig)
    
    # Hotel Search page
    elif selection == "Hotel Search":
        st.title("🔍 Hotel Search")
        st.write("Find hotels anywhere in the world using OpenStreetMap data")
        
        # Create a form for inputs
        with st.form("hotel_search_form"):
            # Location input (required)
            location = st.text_input("Enter a location (e.g. Amsterdam, London, Tokyo)")
            
            # Additional keyword search
            keyword = st.text_input("Filter by name (e.g. Hilton, Marriott)", "")
            
            # Radius slider
            radius = st.slider("Search radius (km)", min_value=1, max_value=10, value=5)
            
            # Submit button
            search_button = st.form_submit_button("Search Hotels")
        
        # Process form submission
        if search_button:
            if not location:
                st.warning("Please enter a location to search for hotels.")
            else:
                with st.spinner(f"Fetching hotels in {location}..."):
                    hotels_found = get_hotels_osm(location, keyword)

                if hotels_found:
                    df = pd.DataFrame(hotels_found)
                    st.success(f"Found {len(df)} hotels in {location}" + 
                              (f" with name containing '{keyword}'" if keyword else ""))
                    
                    # Display dataframe with sorting capability
                    st.dataframe(df, use_container_width=True)
                    
                    # Show map
                    st.subheader("Hotel Locations")
                    try:
                        # Create a map with the hotel locations
                        map_data = pd.DataFrame({
                            'lat': df['lat'].astype(float),
                            'lon': df['lon'].astype(float),
                            'Name': df['Name']
                        })
                        st.map(map_data)
                    except:
                        st.info("Map couldn't be displayed. Detailed location data not available.")
                        
                else:
                    st.warning(f"No hotels found in {location}" + 
                              (f" with name containing '{keyword}'" if keyword else "") + 
                              ". Please try a different location or remove the name filter.")
                    
        # Show sample hotels from dataset
        st.subheader("Or explore our sample hotel dataset:")
        st.dataframe(hotels[['name', 'location', 'price_category', 'avg_rating', 'amenities']])
        
        # Similar hotels
        st.subheader("Find similar hotels:")
        selected_hotel = st.selectbox(
            "Select a hotel to find similar options:", 
            options=hotels['name'].tolist()
        )
        
        if selected_hotel:
            similar_hotels = get_content_recommendations(
                selected_hotel, indices, cosine_sim, hotel_content_df
            )
            
            st.write(f"Hotels similar to {selected_hotel}:")
            st.dataframe(similar_hotels)
    
    # Hotel Recommendations page
    elif selection == "Hotel Recommendations":
        st.title("💎 Personalized Hotel Recommendations")
        st.write("Get hotel recommendations based on your preferences")
        
        # User preferences form
        st.subheader("Tell us your preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            travel_purpose = st.selectbox(
                "What's the purpose of your travel?",
                ["Business", "Leisure", "Family", "Romantic", "Adventure"]
            )
            
            budget = st.selectbox(
                "What's your budget range?",
                ["Budget", "Mid-range", "Premium", "Luxury"]
            )
            
        with col2:
            amenities = st.multiselect(
                "What amenities are important to you?",
                ["WiFi", "Pool", "Spa", "Restaurant", "Gym", "Business Center", 
                 "Beach access", "Room Service", "Kids Club", "Pet-friendly"]
            )
            
            location_pref = st.selectbox(
                "What type of location do you prefer?",
                ["City Center", "Beach", "Mountains", "Countryside", "Any"]
            )
        
        # Create customer profile
        customer_profile = {
            'purpose': travel_purpose,
            'budget': budget,
            'preferred_amenities': amenities,
            'location_preference': location_pref
        }
        
        # Get recommendations button
        if st.button("Get Recommendations"):
            with st.spinner("Finding the perfect hotels for you..."):
                # Get recommended hotels
                recommended_hotels = recommend_hotels(hotels, customer_profile, reviews_analyzed)
                
                # Display top recommendations
                st.subheader("Top Recommended Hotels for You")
                
                if recommended_hotels:
                    for i, hotel in enumerate(recommended_hotels[:5]):
                        with st.container():
                            col1, col2 = st.columns([1, 3])
                            
                            with col1:
                                st.subheader(f"#{i+1}")
                                st.metric("Match Score", f"{hotel['total_score']:.2f}")
                                
                            with col2:
                                st.subheader(hotel['name'])
                                st.write(f"**Location:** {hotel['location']} | **Category:** {hotel['price_category']}")
                                st.write(f"**Rating:** {hotel['avg_rating']}/5")
                                st.write(f"**Amenities:** {hotel['amenities']}")
                                
                                # Show match details
                                st.progress(hotel['total_score'])
                                
                                match_details = {
                                    "Amenities Match": hotel['amenity_score'],
                                    "Rating Score": hotel['rating_score'],
                                    "Purpose Match": hotel['purpose_score'],
                                    "Budget Match": hotel['budget_score']
                                }
                                
                                st.write("**Match Details:**")
