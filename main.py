import streamlit as st
import requests
import pandas as pd
import time
import urllib.parse
import random
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(
    page_title="Smart Hotel Recommender",
    page_icon="🏨",
    layout="wide"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .hotel-card {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
    }
    .recommendation-badge {
        background-color: #FF5722;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        margin-right: 10px;
    }
    .rating-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .rating-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .rating-low {
        color: #F44336;
        font-weight: bold;
    }
    .hotel-image {
        width: 100%;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .feature-tag {
        background-color: #E0F7FA;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8em;
        margin-right: 5px;
    }
</style>
""", unsafe_allow_html=True)

def get_hotels_with_recommendations(location, preferences, budget_level, trip_purpose, min_rating=0):
    # URL encode the location
    encoded_location = urllib.parse.quote(location)
    
    # Use Nominatim for geocoding (finding coordinates from location name)
    geocode_url = f"https://nominatim.openstreetmap.org/search?q={encoded_location}&format=json"
    
    headers = {
        "User-Agent": "HotelRecommenderApp/1.0"  # Nominatim requires a user agent
    }
    
    if st.session_state.get('debug_mode', False):
        st.sidebar.write("Geocoding URL:")
        st.sidebar.code(geocode_url, language="text")
    
    try:
        # Get coordinates for the location
        response = requests.get(geocode_url, headers=headers)
        response.raise_for_status()
        location_data = response.json()
        
        if not location_data:
            if st.session_state.get('debug_mode', False):
                st.sidebar.error("Location not found in OpenStreetMap")
            return []
        
        # Extract latitude and longitude
        lat = location_data[0]["lat"]
        lon = location_data[0]["lon"]
        
        if st.session_state.get('debug_mode', False):
            st.sidebar.success(f"Location found: {location_data[0].get('display_name')}")
            st.sidebar.write(f"Coordinates: {lat}, {lon}")
        
        # Use Overpass API to find hotels
        overpass_url = "https://overpass-api.de/api/interpreter"
        
        # Build more sophisticated query based on preferences
        amenity_filters = []
        
        # Add amenity filters based on preferences
        if "Pool" in preferences:
            amenity_filters.append('["leisure"="swimming_pool"]')
        if "Spa" in preferences:
            amenity_filters.append('["leisure"="spa"]')
        if "Restaurant" in preferences:
            amenity_filters.append('["amenity"="restaurant"]')
        if "Free WiFi" in preferences:
            amenity_filters.append('["internet_access"="wlan"]')
        if "Gym" in preferences:
            amenity_filters.append('["leisure"="fitness_centre"]')
        if "Pet Friendly" in preferences:
            amenity_filters.append('["pets"="yes"]')
            
        # Build the query
        amenity_filter_str = "".join(amenity_filters) if amenity_filters else ""
        
        # Query for hotels within radius (adjusted based on location type - cities get larger radius)
        radius = 10000  # 10km default
        if "city" in location_data[0].get("type", ""):
            radius = 15000  # 15km for cities
        
        # Query for hotels
        overpass_query = f"""
        [out:json];
        (
          node["tourism"="hotel"]{amenity_filter_str}(around:{radius},{lat},{lon});
          way["tourism"="hotel"]{amenity_filter_str}(around:{radius},{lat},{lon});
          node["building"="hotel"]{amenity_filter_str}(around:{radius},{lat},{lon});
          way["building"="hotel"]{amenity_filter_str}(around:{radius},{lat},{lon});
        );
        out body;
        """
        
        if st.session_state.get('debug_mode', False):
            st.sidebar.write("Overpass Query:")
            st.sidebar.code(overpass_query, language="text")
        
        # Respect rate limits
        time.sleep(1)
        
        # Send query
        response = requests.post(overpass_url, data=overpass_query)
        response.raise_for_status()
        hotels_data = response.json()
        
        if st.session_state.get('debug_mode', False):
            st.sidebar.write(f"Found {len(hotels_data.get('elements', []))} hotels")
        
        # Process and enhance hotel data
        hotels = []
        for element in hotels_data.get("elements", []):
            tags = element.get("tags", {})
            
            # Skip hotels without names
            if not tags.get("name"):
                continue
                
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
            
            # Generate synthetic data where OSM data is limited
            # This would be replaced with real data in a production system
            stars = tags.get("stars")
            if not stars:
                stars = random.randint(1, 5)
            else:
                try:
                    stars = int(float(stars))
                except:
                    stars = random.randint(2, 5)
            
            # Generate rating (OSM doesn't have ratings)
            rating = tags.get("rating")
            if not rating:
                # Weight based on stars
                base = 5.0 if stars >= 4 else (4.0 if stars >= 3 else 3.0)
                variation = random.uniform(-0.9, 0.9)
                rating = round(min(5.0, max(1.0, base + variation)), 1)
            else:
                try:
                    rating = float(rating)
                except:
                    rating = round(random.uniform(3.0, 4.9), 1)
            
            # Skip hotels with ratings below minimum threshold
            if rating < min_rating:
                continue
                
            # Generate price level based on stars
            if budget_level == "Budget":
                if stars >= 4:
                    continue  # Skip luxury hotels for budget travelers
                price_range = "€" * random.randint(1, 2)
                price_per_night = random.randint(40, 90)
            elif budget_level == "Mid-range":
                if stars < 2 or stars > 4:
                    continue  # Skip very basic or luxury hotels
                price_range = "€" * random.randint(2, 3)
                price_per_night = random.randint(80, 180)
            else:  # Luxury
                if stars < 3:
                    continue  # Skip basic hotels for luxury travelers
                price_range = "€" * random.randint(3, 5)
                price_per_night = random.randint(150, 450)
            
            # Detect amenities (some from OSM, some generated)
            amenities = []
            if "internet_access" in tags or random.random() > 0.2:
                amenities.append("WiFi")
            if "swimming_pool" in tags or "leisure" in tags and "swimming_pool" in tags["leisure"] or random.random() > 0.7:
                amenities.append("Pool")
            if "restaurant" in tags or random.random() > 0.4:
                amenities.append("Restaurant")
            if "parking" in tags or random.random() > 0.5:
                amenities.append("Parking")
            if "air_conditioning" in tags or random.random() > 0.6:
                amenities.append("A/C")
            if random.random() > 0.7:
                amenities.append("Gym")
            if random.random() > 0.8:
                amenities.append("Spa")
            
            # Calculate appropriate match scores for different trip purposes
            business_score = 0
            family_score = 0
            romantic_score = 0
            solo_score = 0
            
            # Business travel
            if "internet_access" in tags:
                business_score += 2
            if "conference" in tags:
                business_score += 3
            if "fitness_centre" in tags:
                business_score += 1
            if stars >= 3:
                business_score += 2
            if "Pool" in amenities:
                business_score += 1
            
            # Family travel
            if "swimming_pool" in tags:
                family_score += 3
            if stars >= 3 and stars <= 4:
                family_score += 2
            if "Parking" in amenities:
                family_score += 2
            if "Restaurant" in amenities:
                family_score += 2
            
            # Romantic travel
            if stars >= 4:
                romantic_score += 3
            if "Spa" in amenities:
                romantic_score += 2
            if "Restaurant" in amenities:
                romantic_score += 2
            
            # Solo travel
            if "WiFi" in amenities:
                solo_score += 2
            if price_per_night < 120:
                solo_score += 2
            if "Restaurant" in amenities:
                solo_score += 1
            
            # Add random variation
            business_score += random.randint(-1, 2)
            family_score += random.randint(-1, 2)
            romantic_score += random.randint(-1, 2)
            solo_score += random.randint(-1, 2)
            
            # Match score based on selected trip purpose
            if trip_purpose == "Business":
                match_score = business_score
            elif trip_purpose == "Family":
                match_score = family_score
            elif trip_purpose == "Romantic":
                match_score = romantic_score
            else:  # Solo
                match_score = solo_score
                
            # Boost score for each matched preference
            for preference in preferences:
                if preference == "Pool" and "Pool" in amenities:
                    match_score += 2
                elif preference == "Spa" and "Spa" in amenities:
                    match_score += 2
                elif preference == "Restaurant" and "Restaurant" in amenities:
                    match_score += 1
                elif preference == "Free WiFi" and "WiFi" in amenities:
                    match_score += 1
                elif preference == "Gym" and "Gym" in amenities:
                    match_score += 1
            
            # Get approximate distance from center
            if element.get("type") == "node" and element.get("lat") and element.get("lon"):
                lat2 = float(element.get("lat"))
                lon2 = float(element.get("lon"))
                # Simple approximation - not accurate for large distances
                distance = ((float(lat) - lat2)**2 + (float(lon) - lon2)**2)**0.5 * (111.32 * 1000)  # meters
                distance_str = f"{int(distance/1000)} km from center" if distance > 1000 else f"{int(distance)} m from center"
            else:
                distance_str = "Distance unknown"
                distance = 5000  # Default for sorting
                
            # Generate hotel data
            hotel = {
                "Name": tags.get("name"),
                "Address": address,
                "Stars": stars,
                "Rating": rating,
                "Price Range": price_range,
                "Price Per Night": f"${price_per_night}",
                "Raw Price": price_per_night,  # For sorting
                "Phone": tags.get("phone", "Not available"),
                "Website": tags.get("website", "Not available"),
                "Amenities": amenities,
                "Match Score": match_score,
                "Distance": distance_str,
                "Raw Distance": distance,  # For sorting
                "Description": generate_hotel_description(tags.get("name"), stars, trip_purpose, amenities)
            }
            
            hotels.append(hotel)
        
        # Sort by match score (descending)
        hotels.sort(key=lambda x: x["Match Score"], reverse=True)
        
        return hotels
    
    except requests.exceptions.RequestException as e:
        if st.session_state.get('debug_mode', False):
            st.sidebar.error(f"Request Error: {str(e)}")
        return []

def generate_hotel_description(name, stars, trip_purpose, amenities):
    """Generate a synthetic hotel description"""
    descriptions = [
        f"{name} offers comfortable accommodation with {len(amenities)} key amenities including {', '.join(amenities[:3]) if amenities else 'modern facilities'}.",
        f"Located in a convenient area, {name} is a {stars}-star property suitable for {trip_purpose.lower()} travelers.",
        f"Featuring {', '.join(amenities[:2]) if amenities else 'modern amenities'}, {name} provides a pleasant stay experience.",
        f"This {stars}-star establishment delivers quality service and features {', '.join(amenities[:3]) if amenities else 'comfortable accommodations'}.",
        f"{name} welcomes guests with {stars}-star comfort and amenities including {', '.join(amenities[:2]) if amenities else 'essential services'}."
    ]
    return random.choice(descriptions)

def main():
    # Initialize session state
    if 'searched' not in st.session_state:
        st.session_state.searched = False
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    # Create two columns for the header
    header_col1, header_col2 = st.columns([3, 1])
    
    with header_col1:
        st.title("🏨 Smart Hotel Recommender")
        st.write("Find personalized hotel recommendations based on your preferences")
    
    with header_col2:
        # Debug mode toggle in the header right
        debug_toggle = st.checkbox("Show Debug Information", value=st.session_state.debug_mode)
        st.session_state.debug_mode = debug_toggle
    
    # Create a sidebar for search parameters
    st.sidebar.title("Find Your Perfect Stay")
    
    # Create a form for inputs
    with st.sidebar.form("hotel_search_form"):
        # Location input (required)
        location = st.text_input("Destination", "London")
        
        # Trip details
        col1, col2 = st.columns(2)
        with col1:
            check_in = st.date_input("Check-in", datetime.now())
        with col2:
            check_out = st.date_input("Check-out", datetime.now() + timedelta(days=3))
            
        # Trip purpose
        trip_purpose = st.selectbox(
            "Trip Purpose",
            options=["Business", "Family", "Romantic", "Solo"]
        )
        
        # Budget level
        budget_level = st.select_slider(
            "Budget Level",
            options=["Budget", "Mid-range", "Luxury"]
        )
        
        # Preferences
        st.write("Preferences")
        preferences = st.multiselect(
            "Select amenities you prefer",
            options=["Pool", "Spa", "Restaurant", "Free WiFi", "Gym", "Pet Friendly"]
        )
        
        # Minimum rating
        min_rating = st.slider("Minimum Rating", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
        
        # Submit button
        search_button = st.form_submit_button("Find Hotels")
    
    # Process form submission
    if search_button:
        st.session_state.searched = True
        if not location:
            st.warning("Please enter a destination to search for hotels.")
            return

        with st.spinner(f"Finding the best hotels in {location} for your {trip_purpose.lower()} trip..."):
            hotels = get_hotels_with_recommendations(
                location, 
                preferences,
                budget_level,
                trip_purpose,
                min_rating
            )

        if hotels:
            # Display recommendations
            st.subheader(f"🌟 Top Hotel Recommendations for {location}")
            
            # Add filter options
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox(
                    "Sort by",
                    options=["Recommendation Score", "Rating", "Price (Low to High)", "Price (High to Low)", "Distance"],
                    key="sort_by"
                )
            
            with col2:
                filter_amenity = st.multiselect(
                    "Filter by amenities",
                    options=["Pool", "Spa", "Restaurant", "WiFi", "Parking", "Gym", "A/C"],
                    key="filter_amenity"
                )
            
            # Apply sorting
            if sort_by == "Rating":
                hotels.sort(key=lambda x: x["Rating"], reverse=True)
            elif sort_by == "Price (Low to High)":
                hotels.sort(key=lambda x: x["Raw Price"])
            elif sort_by == "Price (High to Low)":
                hotels.sort(key=lambda x: x["Raw Price"], reverse=True)
            elif sort_by == "Distance":
                hotels.sort(key=lambda x: x["Raw Distance"])
            # Default is already sorted by recommendation score
            
            # Apply filtering
            if filter_amenity:
                hotels = [h for h in hotels if all(amenity in h["Amenities"] for amenity in filter_amenity)]
            
            if not hotels:
                st.warning("No hotels match your filtered criteria. Try removing some filters.")
                return
                
            st.success(f"Found {len(hotels)} hotels matching your criteria")
            
            # Create hotel cards
            for i, hotel in enumerate(hotels):
                with st.container():
                    st.markdown(f"""<div class="hotel-card">""", unsafe_allow_html=True)
                    
                    # Create two columns for the hotel info
                    col1, col2 = st.columns([2, 3])
                    
                    with col1:
                        # Use a placeholder image (would be real hotel images in production)
                        hotel_img_num = (i % 5) + 1  # Cycle through 5 placeholder images
                        st.image(f"https://source.unsplash.com/random/300x200/?hotel,room,{hotel_img_num}", 
                                use_column_width=True)
                        
                        # Pricing and booking info
                        st.markdown(f"""
                        <h3>{hotel["Price Per Night"]}<span style="font-size:0.8em;"> per night</span></h3>
                        <p>Total: ${int(hotel["Raw Price"] * (check_out - check_in).days)} for {(check_out - check_in).days} nights</p>
                        """, unsafe_allow_html=True)
                        
                        st.button("Reserve Now", key=f"book_{i}")
                        
                    with col2:
                        # Hotel header with name and recommendation badge
                        if i < 3:  # Top 3 recommendations get special badges
                            badge = f"""<span class="recommendation-badge">Top {i+1} Pick!</span>"""
                        else:
                            badge = ""
                            
                        star_display = "⭐" * hotel["Stars"]
                        
                        st.markdown(f"""
                        <h2>{badge}{hotel["Name"]}</h2>
                        <p>{star_display} · {hotel["Distance"]}</p>
                        """, unsafe_allow_html=True)
                        
                        # Hotel rating with color coding
                        rating_class = "rating-high" if hotel["Rating"] >= 4.5 else ("rating-medium" if hotel["Rating"] >= 3.5 else "rating-low")
                        st.markdown(f"""<p>Rating: <span class="{rating_class}">{hotel["Rating"]}/5</span></p>""", unsafe_allow_html=True)
                        
                        # Description
                        st.markdown(f"""<p>{hotel["Description"]}</p>""", unsafe_allow_html=True)
                        
                        # Amenities as tags
                        amenity_tags = " ".join([f'<span class="feature-tag">{amenity}</span>' for amenity in hotel["Amenities"]])
                        st.markdown(f"""<div>{amenity_tags}</div>""", unsafe_allow_html=True)
                        
                        # Match score for your preferences
                        match_percentage = min(100, int(hotel["Match Score"] * 10))
                        st.progress(match_percentage/100)
                        st.markdown(f"""<p style="text-align:right">{match_percentage}% match for your {trip_purpose.lower()} trip</p>""", unsafe_allow_html=True)
                    
                    st.markdown("""</div>""", unsafe_allow_html=True)
            
            # Add map of hotels
            st.subheader("Hotel Locations")
            st.write("Map functionality would require additional configuration")
            
        else:
            st.warning(f"No hotels found in {location} matching your criteria. Please try different parameters.")
            
            # Provide troubleshooting help
            with st.expander("Troubleshooting Tips"):
                st.markdown("""
                ### Common Issues:
                1. **Location not recognized**: Try a more general location (e.g., "London UK" instead of specific neighborhoods)
                2. **Too many filters**: Reduce your preference requirements
                3. **Rating threshold too high**: Lower the minimum rating
                4. **Budget level restrictions**: Try a different budget level
                """)
    
    # First time instructions
    if not st.session_state.searched:
        # Display welcome information
        st.subheader("🌟 Welcome to Smart Hotel Recommender!")
        st.write("Find your perfect accommodation based on your personal preferences and trip purpose.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### How it works:
            1. Enter your destination and trip details
            2. Select your preferences and budget level
            3. Get personalized hotel recommendations
            4. Filter and sort to find your perfect match
            """)
        
        with col2:
            st.markdown("""
            ### Recommendation factors:
            - ✅ Match for your trip purpose
            - ✅ Amenities that matter to you
            - ✅ Your budget preferences
            - ✅ Hotel ratings and quality
            - ✅ Location and convenience
            """)

if __name__ == "__main__":
    main()
