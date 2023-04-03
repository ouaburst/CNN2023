Jag skickar två filer:

Inlämningsuppgift2_V5_1_Oualid_Burstrom.py: 
Nu importerar den bara numpy och mnist.
Klasserna är däremot kvar. Jag vet att du inte vill ha kod med klasser men jag har försökt omvandla klasserna till enbart definitioner
men det blev rörigt och svårt att följa. Klasserna som jag har implementerat är små med få definitioner.
Koden är dessutom välkommenterad.
När man exekverar koden tar det lång tid att träna modellen.

Inlämningsuppgift2_V5_2_Oualid_Burstrom.py: 
Är den gamla implementationen utan klasser.
Jag har lyckats få till back prop till CNN i träningsloopen.
Men jag får ingen bra resultat.
jag testade med följande parameterarna:
num_epochs = 50
train_images = train_images[:6000]
train_labels = train_labels[:6000]
test_images = test_images[:1000]
test_labels = test_labels[:1000]

Träningen tog extremt lång tid och resultatet är inte imponerande.




