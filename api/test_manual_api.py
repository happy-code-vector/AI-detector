"""
Manual Testing Script for AI Text Detector API (Colab Compatible)

This script allows manual testing with arrays of sentences.
Can be run in Google Colab or locally.

Usage in Colab:
1. Mount Google Drive if needed
2. Update model_path to your model location
3. Run all cells
"""

import sys
import os
from typing import List, Dict, Any
import json

# Add parent directory to path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model service directly
from service_selector import get_model_service


class ManualTesterColab:
    """Manual testing class for Colab usage."""

    def __init__(self, model_path: str = None):
        """Initialize the tester with model path.

        Args:
            model_path: Path to the model directory
        """
        self.service = get_model_service()
        self.model_loaded = False

    def load_model(self, model_path: str = None):
        """Load the AI detection model.

        Args:
            model_path: Optional path to model
        """
        if model_path:
            # Update config if path provided
            from config import settings
            settings.model_path = model_path

        if not self.service.is_loaded:
            print("Loading model...")
            self.service.load_model()
            self.model_loaded = True
            print(f"Model loaded successfully!")
            print(f"Model version: {self.service.model_version}")
            print(f"Device: {self.service.device}")

    def test_sentences(self, sentences: List[str]) -> Dict[str, Any]:
        """Test array of sentences and return detailed analysis.

        Args:
            sentences: List of sentences to test

        Returns:
            Dictionary with detailed analysis
        """
        if not self.model_loaded:
            self.load_model()

        print(f"\nProcessing {len(sentences)} sentence(s)...")

        import time
        start_time = time.time()

        # Get predictions
        probabilities = self.service.predict_batch(sentences)

        processing_time = (time.time() - start_time) * 1000
        total_words = sum(len(s.split()) for s in sentences)

        # Build detailed analysis
        sentences_analysis = []
        for sentence, word_probs in zip(sentences, probabilities):
            words = sentence.split()
            avg_prob = sum(word_probs) / len(word_probs) if word_probs else 0.0

            # Classification based on average probability
            if avg_prob >= 0.6:
                classification = "ai_generated"
            elif avg_prob <= 0.4:
                classification = "human_written"
            else:
                classification = "mixed"

            sentences_analysis.append({
                "sentence": sentence,
                "words": words,
                "probabilities": word_probs,
                "avg_probability": round(avg_prob, 4),
                "classification": classification,
            })

        return {
            "sentences_analysis": sentences_analysis,
            "metadata": {
                "model_version": self.service.model_version or "unknown",
                "processing_time_ms": round(processing_time, 2),
                "total_words": total_words,
            }
        }

    def print_results(self, result: Dict[str, Any]):
        """Print formatted results to console.

        Args:
            result: Result dictionary from test_sentences
        """
        print("\n" + "="*70)
        print("MANUAL TEST RESULTS")
        print("="*70)

        metadata = result.get("metadata", {})
        print(f"\n📊 Metadata:")
        print(f"   Model Version: {metadata.get('model_version')}")
        print(f"   Processing Time: {metadata.get('processing_time_ms')} ms")
        print(f"   Total Words: {metadata.get('total_words')}")

        print(f"\n📝 Sentence Analysis:")
        print("-"*70)

        for i, analysis in enumerate(result.get("sentences_analysis", []), 1):
            sentence = analysis['sentence']
            classification = analysis['classification']
            avg_prob = analysis['avg_probability']

            # Emoji based on classification
            emoji = {
                "ai_generated": "🤖",
                "human_written": "👤",
                "mixed": "🔀"
            }.get(classification, "❓")

            print(f"\n{i}. {emoji} \"{sentence}\"")
            print(f"   Classification: {classification}")
            print(f"   Avg Probability: {avg_prob:.4f}")

            words = analysis.get('words', [])
            probs = analysis.get('probabilities', [])

            if words and probs:
                print(f"   Word-by-Word Analysis:")
                for word, prob in zip(words, probs):
                    bar = self._get_probability_bar(prob)
                    indicator = "🔴" if prob > 0.6 else ("🟢" if prob < 0.4 else "🟡")
                    print(f"     {indicator} {word:15s} | {prob:.4f} {bar}")

        print("\n" + "="*70)

    @staticmethod
    def _get_probability_bar(probability: float, width: int = 20) -> str:
        """Create a visual bar for probability."""
        filled = int(probability * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def export_to_json(self, result: Dict[str, Any], filename: str = "test_results.json"):
        """Export results to JSON file.

        Args:
            result: Result dictionary from test_sentences
            filename: Output filename
        """
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✅ Results exported to {filename}")


# ============================================================================
# USAGE EXAMPLE - Copy this to your Colab notebook
# ============================================================================

def example_usage():
    """Example usage in Colab."""

    # Initialize tester
    tester = ManualTesterColab()

    # Load model (optional - will auto-load on first test)
    # tester.load_model("/path/to/your/model")

    # Test sentences
    test_sentences = [
        "Disney World, Publix Push To Pay Less When Injured Customers Sue\n\nOrlando, FL - July 13, 2013 - Two of Florida's most recognizable corporations, Disney World and Publix Super Markets, are facing criticism for their efforts to limit payouts in lawsuits filed by injured customers.",
        "Amazon Warehouse Injury Rates Spark Federal Investigation\n\nSeattle, WA - April 9, 2024 - Federal safety officials have launched a probe into Amazon after injury reports at fulfillment centers rose 20% in the past year.",
        "Airline Passengers Sue Over Canceled Flights Without Refunds\n\nChicago, IL - October 15, 2025 - Hundreds of travelers claim major carriers refused full refunds after mass cancellations during peak holiday travel.",
        "Tech Company Faces $900M Fine for Data Privacy Violations\n\nSan Francisco, CA - February 20, 2025 - A leading social platform agreed to pay a record penalty for mishandling user location data without consent.",
        "Fast-Food Chain Recalls Chicken Products After Salmonella Outbreak\n\nAtlanta, GA - June 3, 2024 - Several popular items were pulled from menus nationwide following confirmed cases linked to undercooked poultry.",
        "Pharma Giant Settles Lawsuit Over Misleading Painkiller Claims\n\nNew York, NY - August 12, 2024 - The company will pay $1.1 billion to resolve claims it downplayed addiction risks of its opioid medication.",
        "Ride-Sharing Platform Sued for Inadequate Driver Screening\n\nLos Angeles, CA - November 28, 2024 - Plaintiffs allege the company failed to detect criminal histories, leading to multiple passenger assaults.",
        "Retail Chain Accused of Discriminatory Scheduling Practices\n\nDallas, TX - March 17, 2025 - Employees claim the company disproportionately assigned unfavorable shifts to minority workers.",
        "Oil Company Faces $500M Claim After Pipeline Rupture\n\nHouston, TX - January 5, 2025 - Local communities demand compensation for contaminated water and destroyed farmland following a major spill.",
        "Streaming Service Hit with Copyright Lawsuit Over Unlicensed Content\n\nCulver City, CA - September 9, 2025 - Independent creators accuse the platform of streaming their films without proper licensing agreements.",
        "Hotel Group Sued for Failing to Prevent Guest Assaults\n\nMiami, FL - May 22, 2024 - Multiple victims claim poor lighting and absent security contributed to attacks in hotel parking garages.",
        "Automaker Recalls 1.8 Million SUVs Over Steering Failure Risk\n\nDetroit, MI - December 11, 2024 - The company issued an urgent recall after reports of sudden loss of steering control.",
        "Bank Agrees to $185M Settlement for Unauthorized Accounts\n\nCharlotte, NC - April 2, 2025 - The financial institution will compensate customers affected by fraudulent account openings.",
        "Toy Maker Recalls Magnetic Building Sets After Swallowing Incidents\n\nCincinnati, OH - July 19, 2024 - Parents report serious injuries after children ingested small magnets from the popular playset.",
        "Telecom Provider Fined $120M for Deceptive Advertising\n\nDenver, CO - February 14, 2025 - Regulators found the company falsely promised unlimited data while secretly throttling speeds.",
        "Fitness App Sued for Unauthorized Sale of User Health Data\n\nBoston, MA - October 3, 2024 - Users allege sensitive workout and heart-rate information was sold to advertisers without permission.",
        "Construction Firm Faces Lawsuit Over Fatal Scaffold Collapse\n\nPhoenix, AZ - June 28, 2025 - Families of deceased workers claim inadequate safety measures led to the deadly accident.",
        "Grocery Chain Accused of False Organic Labeling\n\nMinneapolis, MN - March 11, 2024 - Customers say conventionally grown produce was marketed as organic at premium prices.",
        "Video Game Publisher Sued Over Addictive Loot Box Mechanics\n\nAustin, TX - November 5, 2024 - Parents claim the system targets children with gambling-like spending features.",
        "Insurance Provider Denies Legitimate Medical Claims\n\nPhiladelphia, PA - January 29, 2025 - A class-action lawsuit alleges systematic rejection of valid treatment requests.",
        "Electric Scooter Operator Faces Pedestrian Injury Lawsuits\n\nPortland, OR - August 20, 2024 - Cities and individuals accuse the company of failing to enforce speed and parking rules.",
        "Cosmetics Brand Recalls Creams Linked to Severe Rashes\n\nLos Angeles, CA - December 2, 2024 - Hundreds of customers reported allergic reactions to a popular skincare line.",
        "Rental Platform Sued for Allowing Discriminatory Listings\n\nNew York, NY - April 18, 2025 - Plaintiffs claim the site enabled landlords to exclude tenants based on race and family status.",
        "Delivery Company Faces Driver Misclassification Lawsuit\n\nSan Diego, CA - February 6, 2025 - Drivers demand employee status and benefits instead of independent contractor classification.",
        "Medical Device Company Settles Over Faulty Heart Valves\n\nMemphis, TN - September 14, 2024 - The firm will pay $380 million to patients who required emergency replacement surgery.",
        "Social Media Platform Fined €250M for Content Moderation Failures\n\nBrussels, Belgium - March 27, 2025 - European regulators penalized repeated delays in removing illegal content.",
        "Restaurant Chain Sued for Systematic Tip Theft\n\nLas Vegas, NV - October 21, 2024 - Servers allege management illegally withheld portions of customer tips.",
        "Automaker Faces Airbag Defect Class Action\n\nNashville, TN - June 9, 2025 - Vehicle owners report airbags failing to deploy during serious collisions.",
        "Online Marketplace Accused of Facilitating Counterfeit Sales\n\nMiami, FL - January 15, 2025 - Luxury brands claim the platform does little to remove fake designer products.",
        "Utility Company Sued Over Extended Blackouts During Storm\n\nNew Orleans, LA - July 2, 2024 - Residents seek damages after week-long power outages during extreme weather.",
        "Treadmill Brand Recalls Models After Injury Reports\n\nSt. Louis, MO - April 23, 2025 - Users report sudden belt acceleration causing falls and fractures.",
        "Credit Card Issuer Settles Hidden Fee Lawsuit\n\nWilmington, DE - November 8, 2024 - The company will refund $165 million to affected cardholders.",
        "Clothing Retailer Faces Forced Arbitration Challenge\n\nPortland, ME - August 31, 2025 - Customers argue mandatory arbitration clauses block legitimate class actions.",
        "Food Delivery App Accused of Tip Diversion\n\nBrooklyn, NY - May 12, 2025 - Drivers claim the platform redirected customer tips to offset operational costs.",
        "Pharmacy Chain Sued for Prescription Overcharges\n\nTampa, FL - February 19, 2024 - Patients allege inflated prices compared to nearby competitors.",
        "Smart TV Manufacturer Settles Patent Dispute\n\nSan Jose, CA - October 26, 2024 - The company agreed to pay $68 million to resolve infringement claims.",
        "Travel Booking Site Fined for Discriminatory Pricing\n\nLondon, UK - March 8, 2025 - Regulators found evidence of location-based price manipulation.",
        "E-commerce Platform Faces Lawsuit Over Fake Reviews\n\nBerlin, Germany - December 14, 2024 - Sellers and buyers claim manipulated ratings distorted purchasing decisions.",
        "Car Rental Company Sued for Unjustified Damage Fees\n\nOrlando, FL - September 17, 2025 - Customers report being charged for pre-existing scratches and dents.",
        "Protein Supplement Brand Recalls Products for Contamination\n\nSalt Lake City, UT - June 25, 2025 - Several flavors were removed after bacterial contamination concerns.",
        "Telehealth Company Settles Data Breach Lawsuit\n\nSan Antonio, TX - January 30, 2025 - The provider will pay $52 million after patient records were exposed.",
        "Home Security Firm Faces False Alarm Litigation\n\nDenver, CO - November 19, 2024 - Customers claim faulty sensors triggered unnecessary emergency responses.",
        "Pet Food Company Recalls Kibble After Illness Reports\n\nKansas City, MO - April 7, 2025 - Multiple brands were pulled following pet sickness complaints.",
        "Ride-Hailing Service Sued Over Emergency Surge Pricing\n\nAustin, TX - July 11, 2024 - Riders allege excessive fares during natural disasters and evacuations.",
        "Banking App Faces Lawsuit After Security Breach\n\nCharlotte, NC - March 4, 2025 - Customers report unauthorized transactions following a data leak.",
        "Furniture Retailer Sued for Chronic Delivery Delays\n\nColumbus, OH - October 8, 2024 - Buyers claim months-long waits and damaged merchandise upon arrival.",
        "Gaming Console Maker Settles Controller Drift Case\n\nRedmond, WA - May 16, 2025 - The company will offer free repairs for affected devices.",
        "Airline Accused of Systematic Overbooking Practices\n\nChicago, IL - December 22, 2024 - Passengers allege confirmed seats were sold to maximize revenue.",
        "Beauty Brand Faces False Advertising Claims\n\nNew York, NY - February 28, 2025 - Plaintiffs say anti-aging claims lacked scientific backing.",
        "Transit Agency Sued for ADA Accessibility Failures\n\nSan Francisco, CA - August 5, 2024 - Advocates claim public transport remains inaccessible to many disabled riders.",
        "Electronics Retailer Settles Price-Fixing Lawsuit\n\nBoston, MA - June 19, 2025 - The chain will pay $140 million for alleged coordinated price increases.",
        "Online Lender Faces Predatory Loan Allegations\n\nLas Vegas, NV - September 30, 2024 - Borrowers claim interest rates and fees violated state usury laws.",
        "Sporting Goods Brand Recalls Helmets After Cracks\n\nPittsburgh, PA - March 13, 2025 - Several football helmet models were pulled due to safety concerns.",
        "Social Network Fined for Child Privacy Violations\n\nWashington, DC - November 25, 2024 - Regulators imposed a $170 million penalty for collecting data from minors.",
        "Grocery Delivery Service Sued Over Spoiled Food\n\nSeattle, WA - January 18, 2025 - Customers report receiving perishable items that were already unsafe to eat.",
        "Solar Panel Installer Faces Defective Product Claims\n\nAlbuquerque, NM - July 7, 2025 - Homeowners allege panels failed prematurely, causing major financial loss.",
        "Streaming Platform Sued for Sudden Content Removal\n\nCulver City, CA - April 29, 2025 - Creators claim their shows were removed without notice or compensation.",
        "Car Dealership Chain Accused of Hidden Fees\n\nPhoenix, AZ - October 12, 2024 - Buyers say dealerships added undisclosed charges at closing.",
        "Diet Supplement Company Recalls Pills for Undeclared Drugs\n\nOrlando, FL - February 3, 2025 - Several weight-loss products contained unlisted pharmaceutical ingredients.",
        "Public Utility Sued for Ignoring Pipeline Warnings\n\nTulsa, OK - May 20, 2025 - Residents claim the company failed to act on known leak risks.",
        "Fitness Tracker Maker Settles Battery Fire Lawsuit\n\nSan Francisco, CA - August 14, 2024 - The company will replace affected devices after overheating incidents.",
        "Online Education Platform Faces Data Breach Suit\n\nAustin, TX - December 9, 2024 - Student records were exposed, prompting privacy violation claims.",
        "Restaurant App Sued for Hidden Subscription Fees\n\nMiami, FL - March 26, 2025 - Users allege automatic charges continued after cancellation attempts.",
        "Construction Equipment Rental Firm Recalls Faulty Scaffolds\n\nCleveland, OH - June 11, 2025 - Several models were pulled after collapse incidents.",
        "Credit Monitoring Service Settles Overcharge Claims\n\nWilmington, DE - September 18, 2024 - Customers will receive refunds for unauthorized monthly fees.",
        "Luxury Watch Brand Sued for Counterfeit Sales\n\nNew York, NY - January 22, 2025 - The company claims online marketplaces failed to stop fake versions.",
        "Ride-Sharing Company Faces Surge Pricing Litigation\n\nDenver, CO - April 15, 2025 - Passengers allege unfair price spikes during public emergencies.",
        "Baby Product Maker Recalls Cribs After Injuries\n\nColumbus, OH - July 29, 2024 - Drop-side mechanisms were blamed for entrapment incidents.",
        "Streaming Music Service Sued for Royalty Underpayment\n\nNashville, TN - November 3, 2024 - Independent artists claim millions in unpaid royalties.",
        "Home Improvement Chain Faces Moldy Lumber Lawsuit\n\nMinneapolis, MN - February 10, 2025 - Customers report purchasing contaminated wood that damaged homes.",
        "Electric Vehicle Maker Recalls Cars Over Battery Risk\n\nPalo Alto, CA - May 27, 2025 - Certain models were pulled due to potential fire hazards.",
        "Online Pharmacy Sued for Delayed Medication Delivery\n\nTampa, FL - August 20, 2024 - Patients claim life-saving prescriptions arrived weeks late.",
        "Gaming Platform Faces Loot Box Gambling Lawsuit\n\nLos Angeles, CA - December 16, 2024 - Parents argue the system violates gambling regulations.",
        "Hotel Booking Site Accused of Fake Listings\n\nLondon, UK - March 4, 2025 - Travelers report non-existent or misrepresented accommodations.",
        "Food Delivery Driver Sues Over Unpaid Mileage\n\nBrooklyn, NY - June 12, 2025 - Drivers claim the platform failed to reimburse vehicle expenses.",
        "Mattress Brand Recalls Beds Due to Chemical Odor\n\nAtlanta, GA - September 25, 2024 - Customers reported persistent off-gassing and health complaints.",
        "Financial App Faces Unauthorized Transaction Lawsuit\n\nCharlotte, NC - January 7, 2025 - Users allege inadequate security allowed account takeovers.",
        "Toy Store Chain Sued for Selling Unsafe Products\n\nChicago, IL - April 18, 2025 - Parents claim recalled items remained on shelves.",
        "Cable Provider Fined for Slow Internet Speeds\n\nDallas, TX - July 23, 2024 - Regulators found widespread failure to deliver advertised bandwidth.",
        "Fitness Center Chain Faces Membership Cancellation Suit\n\nPortland, OR - October 30, 2024 - Members allege excessive barriers to ending contracts.",
        "Smart Home Device Maker Recalls Cameras After Hack\n\nSan Jose, CA - February 14, 2025 - Security flaws allowed unauthorized access to live feeds.",
        "Online Retailer Sued for Misleading Shipping Times\n\nSeattle, WA - May 9, 2025 - Customers claim promised delivery dates were rarely met.",
        "Car Insurance Company Denies Valid Collision Claims\n\nPhoenix, AZ - August 17, 2024 - Drivers allege systematic rejection of legitimate accident reports.",
        "Vitamin Supplement Brand Recalls Pills for Contamination\n\nSalt Lake City, UT - November 21, 2024 - Products were pulled after metal particles were discovered.",
        "Public Transit Operator Sued for Safety Violations\n\nBoston, MA - January 29, 2025 - Riders claim poor maintenance led to derailments and injuries.",
        "Meal Kit Service Faces Spoilage Complaints\n\nNew York, NY - April 5, 2025 - Customers report receiving warm, unsafe ingredients.",
        "Electric Bike Company Recalls Models After Brake Failure\n\nPortland, OR - July 12, 2025 - Several injuries were linked to sudden brake loss.",
        "Cloud Storage Provider Settles Data Loss Lawsuit\n\nAustin, TX - October 19, 2024 - Users claim irreplaceable files disappeared without warning.",
        "Jewelry Retailer Sued for Selling Fake Diamonds\n\nMiami, FL - December 28, 2024 - Customers allege lab-grown stones were sold as natural.",
        "Streaming Device Maker Faces Privacy Violation Claims\n\nSan Francisco, CA - March 15, 2025 - The company allegedly collected viewing habits without consent.",
        "Home Security Company Sued Over Subscription Traps\n\nDenver, CO - June 22, 2025 - Customers claim difficulty canceling automatic renewals.",
        "Pet Supply Brand Recalls Treats After Illness Reports\n\nKansas City, MO - September 8, 2024 - Several dogs became seriously ill after consumption.",
        "Online Lender Accused of Predatory Auto Loans\n\nLas Vegas, NV - November 30, 2024 - Borrowers claim excessive interest rates and hidden fees.",
        "Fitness App Faces Lawsuit Over Inaccurate Heart Data\n\nBoston, MA - February 17, 2025 - Users allege misleading health metrics caused harm.",
        "Construction Material Supplier Sued for Defective Concrete\n\nPhoenix, AZ - May 4, 2025 - Builders report cracking and structural failures.",
        "Video Conferencing Platform Settles Privacy Breach Case\n\nSan Jose, CA - August 11, 2024 - The company will pay $90 million after data exposure.",
        "Grocery Delivery Service Sued for Missing Items\n\nSeattle, WA - October 26, 2024 - Customers claim frequent incomplete or incorrect orders.",
        "Smart Thermostat Maker Recalls Units After Fires\n\nMinneapolis, MN - January 13, 2025 - Several homes reported overheating incidents.",
        "Online Education Provider Faces Refund Dispute Lawsuit\n\nAustin, TX - April 20, 2025 - Students claim courses were misrepresented and refunds denied.",
        "Luxury Car Brand Recalls Vehicles Over Engine Defects\n\nDetroit, MI - July 27, 2024 - Owners report sudden power loss at highway speeds.",
        "Food Processor Recalls Blenders After Blade Detachment\n\nCincinnati, OH - September 14, 2025 - Multiple injuries were linked to flying parts.",
        "Banking Platform Sued for Freezing Accounts Without Notice\n\nCharlotte, NC - November 21, 2024 - Customers claim sudden lockouts caused financial hardship.",
        "Baby Monitor Maker Faces Security Breach Lawsuit\n\nSan Francisco, CA - February 8, 2025 - Hackers allegedly accessed live video feeds of infants."
    ]

    # Run test
    result = tester.test_sentences(test_sentences)

    # Print results
    tester.print_results(result)

    # Export to JSON
    tester.export_to_json(result)

    return result


if __name__ == "__main__":
    # Run example
    result = example_usage()
