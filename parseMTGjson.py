import json

def main():
	color_dict = {"W": 0, "U": 1, "B": 2, "R": 3, "G": 4, "C": 5}
	compressed_card_dict = {}

	count = 0
	with open('AllCards.json') as json_file:
		json_data = json.load(json_file)
		json_file.close()
	print "Total Cards: ", len(json_data.keys())
	for key in json_data.keys():
		card = json_data[key]
		try:
			cardname = card["name"]
			cardtext = card["text"]
			cardtext = "CARDNAME".join(cardtext.split(cardname))
			try: 
				colorIdentity = card["colorIdentity"]
			except KeyError:
				colorIdentity = ["C"]
			target_vector = [0]*6
			for color in colorIdentity:
				target_vector[color_dict[color]] = 1.0
			compressed_card_dict[cardname] = {"text": cardtext, "colors": target_vector}
			count += 1

		except KeyError:
			pass

	print "Total Data Set: ", count

	with open("MTGcardtextcolors.json", 'w') as fp:
		json.dump(compressed_card_dict, fp, sort_keys=True, indent=4)


if __name__ == '__main__':
	main()