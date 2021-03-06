import logging
import numpy as np
import awkward as ak
import json
import copy
from coffea import processor, hist
from coffea.analysis_tools import Weights, PackedSelection
from coffea.lumi_tools import LumiMask
from boostedhiggs.btag import BTagEfficiency, BTagCorrector
from boostedhiggs.common import (
    getBosons,
    bosonFlavor,
    pass_json_array,
)
from boostedhiggs.corrections import (
    corrected_msoftdrop,
    n2ddt_shift,
    powheg_to_nnlops,
    add_pileup_weight,
    add_VJets_NLOkFactor,
    add_VJets_kFactors,
    add_jetTriggerWeight,
    add_jetTriggerSF,
    jet_factory,
    fatjet_factory,
    add_jec_variables,
    met_factory,
    lumiMasks,
)


logger = logging.getLogger(__name__)


def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out


class HbbMuProcessor(processor.ProcessorABC):
    def __init__(self, year='2017', jet_arbitration='pt', btagV2=False,
                 nnlops_rew=False, skipJER=False, tightMatch=False, newVjetsKfactor=True,
        ):

        self._year = year
        self._nnlops_rew = nnlops_rew # for 2018, reweight POWHEG to NNLOPS
        self._jet_arbitration = jet_arbitration
        self._skipJER = skipJER
        self._tightMatch = tightMatch
        self._newVjetsKfactor= newVjetsKfactor

        self._btagV2 = btagV2
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

        self._json_paths = {
            '2016': 'jsons/Cert_271036-284044_13TeV_23Sep2016ReReco_Collisions16_JSON.txt', 
            '2017': 'jsons/Cert_294927-306462_13TeV_EOY2017ReReco_Collisions17_JSON_v1.txt',
            '2018': 'jsons/Cert_314472-325175_13TeV_17SeptEarlyReReco2018ABC_PromptEraD_Collisions18_JSON.txt',
        }

        self._accumulator = processor.dict_accumulator({
            # dataset -> sumw
            'sumw': processor.defaultdict_accumulator(float),
            'cutflow': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('genflavor', 'Gen. jet flavor', [0, 1, 2, 3, 4]),
                hist.Bin('cut', 'Cut index', 15, 0, 15),
#                hist.Bin('msd', r'Jet $m_{sd}$', 22, 47, 201),          
            ),
            'btagWeight': hist.Hist('Events', hist.Cat('dataset', 'Dataset'), hist.Bin('val', 'BTag correction', 50, 0, 3)),
            'muonkin': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('ptmu',r'Muon $p_{T}$ [GeV]',50,0,500),
                hist.Bin('etamu',r'Muon $\eta$',20,0,5),
                hist.Bin('ddb1', r'Jet ddb score', [0, 0.7, 0.89, 1]),
            ),
            'mujetkin': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('eta1',r'Jet $eta$',20,0,5),
                hist.Bin('ddb1', r'Jet ddb score', 25,0,1),
            ),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata['dataset']
        isRealData = not hasattr(events, "genWeight")
        output = self.accumulator.identity()

        if isRealData:
            # Nominal JEC are already applied in data
            output += self.process_shift(events, None)
            return output

        jec_cache = {}
        nojer = "NOJER" if self._skipJER else ""
        fatjets = fatjet_factory[f"{self._year}mc{nojer}"].build(add_jec_variables(events.FatJet, events.fixedGridRhoFastjetAll), jec_cache)
        jets = jet_factory[f"{self._year}mc{nojer}"].build(add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll), jec_cache)
        met = met_factory.build(events.MET, jets, {})

        shifts = [
            ({"Jet": jets, "FatJet": fatjets, "MET": met}, None),
            ({"Jet": jets.JES_jes.up, "FatJet": fatjets.JES_jes.up, "MET": met.JES_jes.up}, "JESUp"),
            ({"Jet": jets.JES_jes.down, "FatJet": fatjets.JES_jes.down, "MET": met.JES_jes.down}, "JESDown"),
            ({"Jet": jets, "FatJet": fatjets, "MET": met.MET_UnclusteredEnergy.up}, "UESUp"),
            ({"Jet": jets, "FatJet": fatjets, "MET": met.MET_UnclusteredEnergy.down}, "UESDown"),
        ]
        if not self._skipJER:
            shifts.extend([
                ({"Jet": jets.JER.up, "FatJet": fatjets.JER.up, "MET": met.JER.up}, "JERUp"),
                ({"Jet": jets.JER.down, "FatJet": fatjets.JER.down, "MET": met.JER.down}, "JERDown"),
            ])
        return processor.accumulate(self.process_shift(update(events, collections), name) for collections, name in shifts)

    def process_shift(self, events, shift_name):
        dataset = events.metadata['dataset']
        isRealData = not hasattr(events, "genWeight")
        selection = PackedSelection()
        weights = Weights(len(events))
        output = self.accumulator.identity()
        if shift_name is None and not isRealData:
            output['sumw'][dataset] += ak.sum(events.genWeight)

        if isRealData:
            trigger = np.zeros(len(events), dtype='bool')
            lumi_mask = lumiMasks[self._year](events.run, events.luminosityBlock)
            for t in self._triggers[self._year]:
                if t in events.HLT.fields:
                    trigger = trigger | events.HLT[t]

            selection.add('trigger', trigger)
            del trigger
        else:
            selection.add('trigger', np.ones(len(events), dtype='bool'))

        if isRealData:
            selection.add('lumimask', lumiMasks[self._year](events.run, events.luminosityBlock))
        else:
            selection.add('lumimask', np.ones(len(events), dtype='bool'))

        if isRealData:
            trigger = np.zeros(len(events), dtype='bool')
            lumi_mask = lumiMasks[self._year](events.run, events.luminosityBlock)
            for t in self._muontriggers[self._year]:
                if t in events.HLT.fields:
                    trigger = trigger | events.HLT[t]
            selection.add('muontrigger', trigger)
        else:
            selection.add('muontrigger', np.ones(len(events), dtype='bool'))
        
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

        ddb = candidatejet.btagDDBvL
        if self._btagV2:
            ddb = candidatejet.btagDDBvLV2

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
                ak.argmax(ddb)
            ]
        else:
            raise RuntimeError("Unknown candidate jet arbitration")

        ddb = candidatejet.btagDDBvL
        if self._btagV2:
            ddb = candidatejet.btagDDBvLV2

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
        selection.add('ddbpass', (ddb >= 0.89))

        jets = events.Jet
        jets = jets[
            (jets.pt > 30.)
            & (abs(jets.eta) < 5.0)
            & jets.isTight
        ]
        # Protect again "empty" arrays [None, None, None...]
        # if ak.sum(candidatejet.phi) == 0.:
        #     return self.accumulator.identity()
        # only consider first 4 jets to be consistent with old framework
        jets = jets[:, :4]
        dphi = abs(jets.delta_phi(candidatejet))
        selection.add('antiak4btagMediumOppHem', ak.max(jets[dphi > np.pi / 2].btagDeepB, axis=1, mask_identity=False) < BTagEfficiency.btagWPs[self._year]['medium'])
        ak4_away = jets[dphi > 0.8]
        selection.add('ak4btagMedium08', ak.max(ak4_away.btagDeepB, axis=1, mask_identity=False) > BTagEfficiency.btagWPs[self._year]['medium'])

        met = events.MET
        selection.add('met', met.pt < 140.)

        goodelectron = (
            (events.Electron.pt > 10)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.cutBased >= events.Electron.LOOSE)
        )
        nelectrons = ak.sum(goodelectron, axis=1)

        goodmuon = (
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.pfRelIso04_all < 0.25)
            & events.Muon.looseId
        )
        nmuons = ak.sum(goodmuon, axis=1)

        goodmuon_cr = (
            (events.Muon.pt > 55)
            & (abs(events.Muon.eta) < 2.1)
            & (events.Muon.pfRelIso04_all < 0.25)
            & events.Muon.looseId
        )
        nmuons_cr = ak.sum(goodmuon_cr, axis=1)
        leadingmuon = ak.firsts(events.Muon[goodmuon_cr])

        ntaus = ak.sum(
            (
                (events.Tau.pt > 20)
                & (abs(events.Tau.eta) < 2.3)
                & events.Tau.idDecayMode
                & (events.Tau.rawIso < 5)
                & (events.Tau.idDeepTau2017v2p1VSjet)
                & ak.all(events.Tau.metric_table(events.Muon[goodmuon]) > 0.4, axis=2)
                & ak.all(events.Tau.metric_table(events.Electron[goodelectron]) > 0.4, axis=2)
            ),
            axis=1,
        )

        selection.add('noleptons', (nmuons == 0) & (nelectrons == 0) & (ntaus == 0))
        selection.add('noetau', (nelectrons == 0) & (ntaus == 0))
        selection.add('onemuon', (nmuons_cr == 1))
        selection.add('muonDphiAK8', abs(leadingmuon.delta_phi(candidatejet)) > 2*np.pi/3)

        if isRealData :
            genflavor = candidatejet.pt - candidatejet.pt  # zeros_like
        else:
            weights.add('genweight', events.genWeight)
            add_pileup_weight(weights, events.Pileup.nPU, self._year, dataset)
            bosons = getBosons(events.GenPart)
            matchedBoson = candidatejet.nearest(bosons, axis=None, threshold=0.8)
            if self._tightMatch:
                match_mask = ((candidatejet.pt - matchedBoson.pt)/matchedBoson.pt < 0.5) & ((candidatejet.msdcorr - matchedBoson.mass)/matchedBoson.mass < 0.3)
                selmatchedBoson = ak.mask(matchedBoson, match_mask)
                genflavor = bosonFlavor(selmatchedBoson)
            else:
                genflavor = bosonFlavor(matchedBoson)
            genBosonPt = ak.fill_none(ak.firsts(bosons.pt), 0)

            if self._newVjetsKfactor:
                add_VJets_kFactors(weights, events.GenPart, dataset)
            else:
                add_VJets_NLOkFactor(weights, genBosonPt, self._year, dataset)
            add_jetTriggerSF(weights, ak.firsts(fatjets), self._year)

            if shift_name is None:
                output['btagWeight'].fill(dataset=dataset, val=self._btagSF.addBtagWeight(weights, ak4_away))
            logger.debug("Weight statistics: %r" % weights.weightStatistics)

        msd_matched = candidatejet.msdcorr * self._msdSF[self._year] * (genflavor > 0) + candidatejet.msdcorr * (genflavor == 0)

        regions = {
            'signal': ['trigger','lumimask','minjetkin','jetid','jetacceptance','n2ddt','antiak4btagMediumOppHem','met','noleptons'],
            'muoncontrol': ['muontrigger', 'minjetkin_muoncr', 'jetid', 'n2ddt', 'ak4btagMedium08', 'noetau', 'onemuon', 'muonDphiAK8'],
        }

        def normalize(val, cut):
            if cut is None:
                ar = ak.to_numpy(ak.fill_none(val, np.nan))
                return ar
            else:
                ar = ak.to_numpy(ak.fill_none(val[cut], np.nan))
                return ar

        if shift_name is None:
            for region, cuts in regions.items():
                allcuts = set([])
                cut = selection.all(*allcuts)
                output['cutflow'].fill(dataset=dataset, region=region, genflavor=normalize(genflavor, None),
                                    cut=0, weight=weights.weight())#, msd=normalize(msd_matched, None))           
                for i, cut in enumerate(cuts + ['ddbpass']):
                    allcuts.add(cut)
                    cut = selection.all(*allcuts)
                    output['cutflow'].fill(dataset=dataset, region=region, genflavor=normalize(genflavor, cut),
                                        cut=i + 1, weight=weights.weight()[cut])#, msd=normalize(msd_matched, cut))

        if shift_name is None:
            systematics = [None] + list(weights.variations)
        else:
            systematics = [shift_name]

        def fill(region, systematic, wmod=None):
            selections = regions[region]
            cut = selection.all(*selections)
            sname = 'nominal' if systematic is None else systematic
            if wmod is None:
                if systematic in weights.variations:
                    weight = weights.weight(modifier=systematic)[cut]
                else:
                    weight = weights.weight()[cut]
            else:
                weight = weights.weight()[cut] * wmod[cut]
            
            if sname == 'nominal':
                output['muonkin'].fill(
                    dataset=dataset,
                    region=region,
                    ptmu=normalize(leadingmuon.pt, cut),
                    etamu=normalize(abs(leadingmuon.eta),cut),
                    ddb1=normalize(ddb, cut),
                    weight=weight,
                )
                output['mujetkin'].fill(
                    dataset=dataset,
                    region=region,
                    eta1=normalize(abs(candidatejet.eta),cut),
                    ddb1=normalize(ddb, cut),
                    weight=weight,
                )

        for region in regions:
            for systematic in systematics:
                if isRealData and systematic is not None:
                    continue
                fill(region, systematic)
#            if shift_name is None and 'GluGluH' in dataset and 'LHEWeight' in events.fields:
#                for i in range(9):
#                    fill(region, 'LHEScale_%d' % i, events.LHEScaleWeight[:, i])
#                for c in events.LHEWeight.fields[1:]:
#                    fill(region, 'LHEWeight_%s' % c, events.LHEWeight[c])

        if shift_name is None:
            output["weightStats"] = weights.weightStatistics
        return output

    def postprocess(self, accumulator):
        return accumulator
