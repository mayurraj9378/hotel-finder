import streamlit as st
import requests
import pandas as pd
import time
import urllib.parse

def get_hotels_osm(location, keyword=None):
    # URL encode the location
    encoded_location = urllib.parse.quote(location)
    
    # Use Nominatim for geocoding (finding coordinates from location name)
    geocode_url = f"https://nominatim.openstreetmap.org/search?q={encoded_location}&format=json"
    
    headers = {
        "User-Agent": "HotelFinderApp/1.0"  # Nominatim requires a user agent
    }
    
    # Debug info
    st.sidebar.write("Geocoding URL:")
    st.sidebar.code(geocode_url, language="text")
    
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
        
        st.sidebar.success(f"Location found: {location_data[0].get('display_name')}")
        st.sidebar.write(f"Coordinates: {lat}, {lon}")
        
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
        
        # Debug info
        st.sidebar.write("Overpass Query:")
        st.sidebar.code(overpass_query, language="text")
        
        # Respect rate limits
        time.sleep(1)
        
        # Send query
        response = requests.post(overpass_url, data=overpass_query)
        response.raise_for_status()
        hotels_data = response.json()
        
        # Debug info
        st.sidebar.write(f"Found {len(hotels_data.get('elements', []))} hotels")
        
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
            
            # Get hotel details
            hotel = {
                "Name": tags.get("name", "Unnamed Hotel"),
                "Address": address,
                "Stars": tags.get("stars", "N/A"),
                "Phone": tags.get("phone", "N/A"),
                "Website": tags.get("website", "N/A")
            }
            
            hotels.append(hotel)
        
        return hotels
    
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Request Error: {str(e)}")
        return []

def main():
    st.title("🏨 Hotel Finder")
    st.write("Find hotels anywhere in the world using OpenStreetMap data")
    
    # Add debug toggle
    show_debug = st.sidebar.checkbox("Show Debug Information")
    if not show_debug:
        st.sidebar.empty()
    
    # Create a form for inputs
    with st.form("hotel_search_form"):
        # Location input (required)
        location = st.text_input("Enter a location (e.g. Amsterdam, London, Tokyo)")
        
        # Additional keyword search
        keyword = st.text_input("Filter by name (e.g. Hilton, Marriott)", "")
        
        # Radius slider (for future implementation)
        radius = st.slider("Search radius (km)", min_value=1, max_value=10, value=5)
        
        # Submit button
        search_button = st.form_submit_button("Search Hotels")
    
    # Process form submission
    if search_button:
        if not location:
            st.warning("Please enter a location to search for hotels.")
            return

        with st.spinner(f"Fetching hotels in {location}..."):
            hotels = get_hotels_osm(location, keyword)

        if hotels:
            df = pd.DataFrame(hotels)
            st.success(f"Found {len(df)} hotels in {location}" + 
                      (f" with name containing '{keyword}'" if keyword else ""))
            
            # Display dataframe with sorting capability
            st.dataframe(df, use_container_width=True)
            
            # Show map if location found
            st.subheader("Hotel Area")
            try:
                # Create a simple map centered on the location
                map_location = f"{location}" + (f" {keyword}" if keyword else "")
                st.map(pd.DataFrame({
                    'lat': [float(hotels[0].get('lat', 0))],
                    'lon': [float(hotels[0].get('lon', 0))]
                }))
            except:
                st.info("Map couldn't be displayed. Detailed location data not available.")
                
        else:
            st.warning(f"No hotels found in {location}" + 
                      (f" with name containing '{keyword}'" if keyword else "") + 
                      ". Please try a different location or remove the name filter.")
            
            # Provide troubleshooting help
            with st.expander("Troubleshooting Tips"):
                st.markdown("""
                ### Common Issues:
                1. **Location not found**: Try a more general location (e.g., "London UK" instead of specific neighborhoods)
                2. **No hotels in database**: Some areas may have limited OpenStreetMap data
                3. **Name filter too specific**: Remove name filter to see more results
                4. **Network issues**: Check your internet connection
                5. **API rate limits**: If you make too many requests, try again later
                """)

if __name__ == "__main__":
    main()