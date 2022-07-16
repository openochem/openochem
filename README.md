# Open OCHEM -- AI models for drug discovery and enviromental chemistry

The Open OCHEM is open source version of the On-line Chemical database Modelling and Environment Platform (http://ochem.eu)

It is a user-contributed repository of referenced experimental data, computational tools and models of ADMET properties of chemical compounds. The OCHEM algorithms can reliably identify compounds predicted with experimental accuracy: there is no need to test them in a lab. The OCHEM can be used for timely and low-cost identification of scaffolds with lower risks of failure due to the unfavorable physico-chemical and/or biological properties. The free open source of OCHEM is a reference system for academic users thus accumulating data and knowledge produced in academia. The developed OCHEM workflow allows an unbiased comparison of different existing and new machine learning algorithms which can be easily integrated in OCHEM by its users.
	
OCHEM software can be used to develop QSPR and QSAR models for various biological and physico-chemical projects. It can work with millions of molecules and can be configured to use hundrends of CPUs or GPUs. Open OCHEM allows you to install the fully functional version of the software and analyse your data privately. The closed source version is also available from BIGCHEM GmBH and provides several additional optimized software packages which were contributed by the company or its partners.

The open OCHEM currently supports tens methods and descriptors packages, which were developed and contributed by different providers and are distributed under the open source or respective license agreements (most of them are free of charge for academic, educational, recreational or evaluation purposes - check each respective license agreement).

See [installation instructions](./INSTRUCTIONS_OCHEM) how to install and run open the OCHEM.

We wish you a happy computing! 

We sincerely thank Yuriy Sushko, Sergey Novotarskyi, Pavel Karpov, Mark Embrechts, Robert Körner, Anil Kumar Pandey, Elena Salmina, Stefan Brandmaier, Larisa Charochkina, Vasyl Kovalishyn, Ahmed Abdelaziz, Matthias Rupp, Dipan Ghosh, Zhonghua Xia, Alli Keys as well as many other current and former members of Tetko's group and eADMET and BIGCHEM GmbH companies for their contributions to the development, testing, use and the feedback.

We also thank developers of [CDK](github.com/cdk), [MOPAC2016](http://openmopac.net), [KGCNN](https://github.com/aimat-lab/gcnn_keras), [OpenBabel](https://github.com/openbabel/openbabel), [Xemistry](https://www.xemistry.com/), [BALLOON](http://users.abo.fi/mivainio/balloon/), [WEKA](https://github.com/Waikato/weka-3.8)  as well as Vsevolod Tanchuk, Sergey Sosnin, Maxim Fedorov, Peter Ertl, Bruno Bienfait, Ruud van Deursen, Gilles Marcou, Igor Baskin, Artem Cherkasov, Pavel Polishchuk, Eugene Radchenko, Vladimir Palyulin, Vijay Masand, Vishweh Venkatraman, Andrea Mauri, Weida Tong, Huixiao Hong, Todd Martin, Peter Jarowski, Vladimir Poroikov, Dmitriy Filimonov, Atif Raza  and many others who contributed modules that are used in the OCHEM.

Igor Tetko, Martin Šícho, Guillaume Godin and BIGCHEM GmBH



# open OCHEM history:

The OCHEM project started in 2007 following a [GO-Bio award](https://www.go-bio.de/gobio/de/gefoerderte-projekte/_documents/die-toxizitaet-von-wirkstoffen-und-chemikalien-berechnen.html) to Igor Tetko at Helmholtz Munich (HMGU). Between 2011 and 2014 software was licensed and developed by startup [eADMET GmBH company](http://eadmet.com) company. Since 2016 OCHEM is maintained, developed and supported by [BIGCHEM GmBH](http://bigchem.de), HMGU as well by Firmenich, which contributed several important modules. In 2021 as result of agreement between HMGU, BIGCHEM GmBH and Firmenich it was decided to release the core code of OCHEM as an open source under the AGPL v. 3.0 agreement.

The initial version of OCHEM was described in [Tetko et al 2011](https://link.springer.com/article/10.1007/s10822-011-9440-2) and article about openOCHEM is under preparation now. The OCHEM includes superserver, metaserver (both running as independent tomcat instances) as well as calculation servers. It setup requires installation of mariadb, mongodb, singularity as well as several other packages and it is described in details in the [installation instructions](./INSTRUCTIONS_OCHEM). The detailed instructions how to setup OCHEM for maximal performance as well as to connect it to main OCHEM web site will be also provided.
