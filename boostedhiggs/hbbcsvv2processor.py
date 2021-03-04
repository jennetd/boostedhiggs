import logging
import numpy as np
import awkward1 as ak
import json
from coffea import processor, hist
from coffea.analysis_tools import Weights, PackedSelection
from boostedhiggs.btagcsvv2 import BTagEfficiency, BTagCorrector
from boostedhiggs.common import (
    getBosons,
    bosonFlavor,
)
from boostedhiggs.corrections import (
    corrected_msoftdrop,
    n2ddt_shift,
    add_pileup_weight,
    add_VJets_NLOkFactor,
    add_jetTriggerWeight,
)


logger = logging.getLogger(__name__)


class HbbCSVv2Processor(processor.ProcessorABC):
    def __init__(self, year='2017', jet_arbitration='pt'):
        self._year = year
        self._jet_arbitration = jet_arbitration

        self._btagSF = BTagCorrector(year, 'medium')

        self._msdSF = {
            '2016': 1.,
            '2017': 0.987,
            '2018': 0.970,
        }

        with open('muon_triggers.json') as f:
            self._muontriggers = json.load(f)

        with open('triggers.json') as f:
            self._triggers = json.load(f)

        self._accumulator = processor.dict_accumulator({
            # dataset -> sumw
            'sumw': processor.defaultdict_accumulator(float),
            'cutflow': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]),
                hist.Bin('cut', 'Cut index', 15, 0, 15),
            ),
            'btagWeight': hist.Hist('Events', hist.Cat('dataset', 'Dataset'), hist.Bin('val', 'BTag correction', 50, 0, 3)),
            'templates1': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('pt1', r'Jet $p_{T}$ [GeV]', [450, 500, 550, 600, 675, 800, 1200]),
                hist.Bin('msd1', r'Jet $m_{sd}$', 22, 47, 201),
                hist.Bin('ddb1', r'Jet ddb score', [0, 0.89, 1]),
            ),
            'templates2': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ptmu',r'Muon $p_{T}$ [GeV]',100,0,2000),
                hist.Bin('etamu',r'Muon $\eta$',20,0,3),
                hist.Bin('msd1', r'Jet $m_{sd}$', 22, 47, 201),
                hist.Bin('ddb1', r'Jet ddb score', [0, 0.89, 1]),
            ),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata['dataset']
        isRealData = not hasattr(events, "genWeight")
        selection = PackedSelection()
        weights = Weights(len(events))
        output = self.accumulator.identity()
        if not isRealData:
            output['sumw'][dataset] += ak.sum(events.genWeight)

        if isRealData:
            trigger = np.zeros(len(events), dtype='bool')
            for t in self._triggers[self._year]:
                try:
                    trigger = trigger | events.HLT[t]
                except:
                    print("No trigger " + t + " in file")
        else:
            trigger = np.ones(len(events), dtype='bool')
        selection.add('trigger', trigger)

        if isRealData:
            trigger = np.zeros(len(events), dtype='bool')
            for t in self._muontriggers[self._year]:
                try:
                    trigger = trigger | events.HLT[t]
                except:
                    print("No trigger " + t + " in file")
        else:
            trigger = np.ones(len(events), dtype='bool')
        selection.add('muontrigger', trigger)

        fatjets = events.FatJet
        fatjets['msdcorr'] = corrected_msoftdrop(fatjets)
        fatjets['qcdrho'] = 2 * np.log(fatjets.msdcorr / fatjets.pt)
        fatjets['n2ddt'] = fatjets.n2b1 - n2ddt_shift(fatjets, year=self._year)
        fatjets['msdcorr_full'] = fatjets['msdcorr'] * self._msdSF[self._year]

        candidatejet = fatjets[
            # https://github.com/DAZSLE/BaconAnalyzer/blob/master/Analyzer/src/VJetLoader.cc#L269
            (fatjets.pt > 200)
            & (abs(fatjets.eta) < 2.5)
            & fatjets.isTight  # this is loose in sampleContainer
        ]
        if self._jet_arbitration == 'pt':
            candidatejet = ak.firsts(candidatejet)
        elif self._jet_arbitration == 'mass':
            candidatejet = candidatejet[
                ak.argmax(candidatejet.msdcorr)
            ]
        elif self._jet_arbitration == 'n2':
            candidatejet = candidatejet[
                ak.argmin(candidatejet.n2ddt)
            ]
        elif self._jet_arbitration == 'ddb':
            candidatejet = candidatejet[
                ak.argmax(candidatejet.btagDDBvL)
            ]
        else:
            raise RuntimeError("Unknown candidate jet arbitration")

        selection.add('minjetkin',
            (candidatejet.pt >= 450)
            & (candidatejet.msdcorr >= 47.)
            & (abs(candidatejet.eta) < 2.5)
        )
        selection.add('minjetkin_muoncr',
            (candidatejet.pt >= 400)
            & (candidatejet.msdcorr >= 40.)
            & (abs(candidatejet.eta) < 2.5)
        )
        selection.add('jetacceptance',
            (candidatejet.msdcorr >= 40.)
            & (candidatejet.pt < 1200)
            & (candidatejet.msdcorr < 201.)
        )
        selection.add('jetid', candidatejet.isTight)
        selection.add('n2ddt', (candidatejet.n2ddt < 0.))
        selection.add('ddbpass', (candidatejet.btagDDBvL >= 0.89))

        jets = events.Jet[
            (events.Jet.pt > 30.)
            & (abs(events.Jet.eta) < 2.5)
            & events.Jet.isTight
        ]
        # only consider first 4 jets to be consistent with old framework
        jets = jets[:, :4]
        dphi = abs(jets.delta_phi(candidatejet))
        selection.add('antiak4btagMediumOppHem', ak.max(jets[dphi > np.pi / 2].btagCSVV2, axis=1, mask_identity=False) < BTagEfficiency.btagWPs[self._year]['medium'])
        ak4_away = jets[dphi > 0.8]
        selection.add('ak4btagMedium08', ak.max(ak4_away.btagCSVV2, axis=1, mask_identity=False) > BTagEfficiency.btagWPs[self._year]['medium'])

        selection.add('met', events.MET.pt < 140.)

        goodmuon = (
            (events.Muon.pt > 55)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.pfRelIso04_all < 0.25)
            & events.Muon.looseId
            & (abs(events.Muon.delta_phi(candidatejet)) > 2*np.pi/3)
        )
        candidatemuon = ak.firsts(events.Muon[goodmuon])
        ngoodmuons = ak.sum(goodmuon,axis = 1)

        nelectrons = ak.sum(
            (events.Electron.pt > 10.)
            & (abs(events.Electron.eta) < 2.5) 
            & (events.Electron.cutBased >= events.Electron.VETO),
            axis = 1,
        )
        nmuons = ak.sum(
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.pfRelIso04_all < 0.25)
            & events.Muon.looseId,
            axis = 1,
        )
        ntaus = ak.sum(
            (events.Tau.pt > 20.)
            & (events.Tau.idDecayMode)
            & (events.Tau.rawIso < 5)
            & (abs(events.Tau.eta) < 2.3)
            & (events.Tau.idMVAoldDM2017v1 >= 16),
            axis = 1,
        )

        selection.add('noleptons', (nmuons == 0) & (nelectrons == 0) & (ntaus == 0))
        selection.add('noetau', (nelectrons == 0) & (ntaus == 0))
        selection.add('onemuon', (ngoodmuons == 1))
#        selection.add('muonkin', ak.any((candidatemuon.pt > 55.) & (abs(candidatemuon.eta) < 2.1), axis=1))
#        selection.add('muonDphiAK8', ak.any(abs(candidatemuon.delta_phi(candidatejet)) > 2*np.pi/3, axis=1))

        if isRealData:
            genflavor = np.zeros(len(events))
        else:
            weights.add('genweight', events.genWeight)
            add_pileup_weight(weights, events.Pileup.nPU, self._year, dataset)
            bosons = getBosons(events.GenPart)
            matchedBoson = candidatejet.nearest(bosons, axis=None, threshold=0.8)
            genflavor = bosonFlavor(matchedBoson)
            genBosonPt = ak.fill_none(ak.firsts(bosons.pt), 0)
            add_VJets_NLOkFactor(weights, genBosonPt, self._year, dataset)
            add_jetTriggerWeight(weights, candidatejet.msdcorr, candidatejet.pt, self._year)
            output['btagWeight'].fill(dataset=dataset, val=self._btagSF.addBtagWeight(weights, ak4_away))
            logger.debug("Weight statistics: %r" % weights.weightStatistics)

        msd_matched = candidatejet.msdcorr * self._msdSF[self._year] * (genflavor > 0) + candidatejet.msdcorr * (genflavor == 0)

        regions = {
            'signal': ['trigger', 'minjetkin', 'jetacceptance', 'jetid', 'n2ddt', 'antiak4btagMediumOppHem', 'met', 'noleptons'],
            'muoncontrol': ['muontrigger', 'minjetkin_muoncr', 'jetid', 'n2ddt', 'ak4btagMedium08', 'noetau','onemuon'],
            'noselection': [],
        }

        for region, cuts in regions.items():
            allcuts = set()
            output['cutflow'].fill(dataset=dataset, region=region, genflavor=genflavor, cut=0, weight=weights.weight())
            for i, cut in enumerate(cuts + ['ddbpass']):
                allcuts.add(cut)
                cut = selection.all(*allcuts)
                output['cutflow'].fill(dataset=dataset, region=region, genflavor=genflavor[cut], cut=i + 1, weight=weights.weight()[cut])

        systematics = [
            None,
#            'jet_triggerUp',
#            'jet_triggerDown',
#            'btagWeightUp',
#            'btagWeightDown',
#            'btagEffStatUp',
#            'btagEffStatDown',
        ]

        def normalize(val, cut):
            return ak.to_numpy(ak.fill_none(val[cut], np.nan))

        def fill(region):
            selections = regions[region]
            cut = selection.all(*selections)
            weight = weights.weight()[cut]

            output['templates1'].fill(
                dataset=dataset,
                region=region,
                pt1=normalize(candidatejet.pt, cut),
                msd1=normalize(msd_matched, cut),
                ddb1=normalize(candidatejet.btagDDBvL, cut),
                weight=weight,
            )
            output['templates2'].fill(
                dataset=dataset,
                region=region,
                ptmu=normalize(candidatemuon.pt, cut),
                etamu=normalize(abs(candidatemuon.eta),cut),
                msd1=normalize(msd_matched, cut),
                ddb1=normalize(candidatejet.btagDDBvL, cut),
                weight=weight,
            )

        for region in regions:
            fill(region)

        output["weightStats"] = weights.weightStatistics
        return output

    def postprocess(self, accumulator):
        return accumulator