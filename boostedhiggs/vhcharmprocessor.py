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
    add_jetTriggerWeight,
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


class VHCharmProcessor(processor.ProcessorABC):
    def __init__(self, year='2017', jet_arbitration='ddcvb', v2=False, v3=False, v4=False,
            nnlops_rew=False,  skipJER=False, tightMatch=False,
        ):
        # v2 DDXv2
        # v3 ParticleNet
        # v4 mix
        self._year = year
        self._v2 = v2
        self._v3 = v3
        self._v4 = v4
        self._nnlops_rew = nnlops_rew # for 2018, reweight POWHEG to NNLOPS
        self._jet_arbitration = jet_arbitration
        self._skipJER = skipJER
        self._tightMatch = tightMatch

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
            'templates': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Cat('systematic', 'Systematic'),
                hist.Bin('ddb1', r'Jet 1 ddb score', [0, 0.89, 1]),
                hist.Bin('pt1', r'Jet 1 $p_{T}$ [GeV]', [450, 500, 550, 600, 675, 800, 1200]),
                hist.Bin('msd1', r'Jet 1 $m_{sd}$', 22, 47, 201),
                hist.Bin('msd2', r'Jet 2 $m_{sd}$', 22, 47, 201),
            ),
            'templates-vh-1': hist.Hist(
                'Events',
                hist.Cat('dataset', 'Dataset'),
                hist.Cat('region', 'Region'),
                hist.Bin('msd1', r'Jet 1 $m_{sd}$', 22, 47, 201),
                hist.Bin('ddb1', r'Jet 1 ddb score', [0, 0.89, 1]),
                hist.Bin('pt1', r'Jet 1 $p_{T}$ [GeV]', [400, 450, 500, 550, 600, 675, 800, 1200]),
                hist.Bin('msd2', r'Jet 2 $m_{sd}$', 22, 47, 201),
                hist.Bin('ddb2', r'Jet 2 ddb score', [0, 0.89, 1]),
                hist.Bin('pt2', r'Jet 2 $p_{T}$ [GeV]', [400, 450, 500, 550, 600, 675, 800, 1200]),
#                hist.Bin('DR', r'$\Delta $$',8,0,8),
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
        else:
            trigger = np.ones(len(events), dtype='bool')
            lumi_mask  = np.ones(len(events), dtype='bool')
        selection.add('trigger', trigger)
        selection.add('lumimask', lumi_mask)

        if isRealData:
            trigger = np.zeros(len(events), dtype='bool')
            lumi_mask = lumiMasks[self._year](events.run, events.luminosityBlock)
            for t in self._muontriggers[self._year]:
                if t in events.HLT.fields:
                    trigger = trigger | events.HLT[t]
        else:
            trigger = np.ones(len(events), dtype='bool')
            lumi_mask  = np.ones(len(events), dtype='bool')
        selection.add('muontrigger', trigger)
        selection.add('lumimask', lumi_mask)

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
            secondjet = ak.firsts(candidatejet[:, 1:2])
            candidatejet = ak.firsts(candidatejet)

        elif self._jet_arbitration == 'ddcvb':
            
            leadingjets = candidatejet[:, 0:2]
            # ascending = true
            indices = ak.argsort(leadingjets.btagDDCvBV2,axis=1)

            # candidate jet is more b-like
            candidatejet = ak.firsts(leadingjets[indices[:, 0:1]])
            # second jet is more charm-like
            secondjet = ak.firsts(leadingjets[indices[:, 1:2]])

        else:
            raise RuntimeError("Unknown candidate jet arbitration")

        selection.add('jet1kin',
            (candidatejet.pt >= 450)
            & (candidatejet.msdcorr >= 47.)
            & (abs(candidatejet.eta) < 2.5)
        )
        selection.add('jet2kin',
            (secondjet.pt >= 400)
            & (secondjet.msdcorr >= 47.)
            & (abs(secondjet.eta) < 2.5)
        )
        selection.add('minjetkin_muoncr',
            (candidatejet.pt >= 400)
            & (candidatejet.msdcorr >= 40.)
            & (abs(candidatejet.eta) < 2.5)
            & (secondjet.pt >= 400)
            & (secondjet.msdcorr >= 40.)
            & (abs(secondjet.eta) < 2.5)
        )

        selection.add('jetacceptance',
            (candidatejet.msdcorr >= 40.)
            & (candidatejet.pt < 1200)
            & (candidatejet.msdcorr < 201.)
            & (secondjet.msdcorr >= 40.)
            & (secondjet.pt < 1200)
            & (secondjet.msdcorr < 201.)
        )
        selection.add('jetid', 
                      candidatejet.isTight
                      & secondjet.isTight
        )
        selection.add('n2ddt', 
                      (candidatejet.n2ddt < 0.)
                      & (secondjet.n2ddt < 0.)
        )
        selection.add('ddbpass', (candidatejet.btagDDBvL >= 0.89))

        DR = candidatejet.delta_r(secondjet)

        jets = events.Jet
        jets = jets[
            (jets.pt > 30.)
            & (abs(jets.eta) < 2.5)
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
            genflavor2 = secondjet.pt - secondjet.pt
        else:
            weights.add('genweight', events.genWeight)
            add_pileup_weight(weights, events.Pileup.nPU, self._year, dataset)
            bosons = getBosons(events.GenPart)
            matchedBoson = candidatejet.nearest(bosons, axis=None, threshold=0.8)
            matchedBoson2 = secondjet.nearest(bosons, axis=None, threshold=0.8)
            if self._tightMatch:
                match_mask = ((candidatejet.pt - matchedBoson.pt)/matchedBoson.pt < 0.5) & ((candidatejet.msdcorr - matchedBoson.mass)/matchedBoson.mass < 0.3)
                selmatchedBoson = ak.mask(matchedBoson, match_mask)
                genflavor = bosonFlavor(selmatchedBoson)

                match_mask2 = ((secondjet.pt - matchedBoson2.pt)/matchedBoson2.pt < 0.5) & ((secondjet.msdcorr - matchedBoson2.mass)/matchedBoson2.mass < 0.3)
                selmatchedBoson2 = ak.mask(matchedBoson2, match_mask2)
                genflavor2 = bosonFlavor(selmatchedBoson2)
            else:
                genflavor = bosonFlavor(matchedBoson)
                genflavor2 = bosonFlavor(matchedBoson2)
            genBosonPt = ak.fill_none(ak.firsts(bosons.pt), 0)
            add_VJets_NLOkFactor(weights, genBosonPt, self._year, dataset)
            add_jetTriggerWeight(weights, candidatejet.msdcorr, candidatejet.pt, self._year)
            if shift_name is None:
                output['btagWeight'].fill(dataset=dataset, val=self._btagSF.addBtagWeight(weights, ak4_away))

        msd_matched = candidatejet.msdcorr * self._msdSF[self._year] * (genflavor > 0) + candidatejet.msdcorr * (genflavor == 0)
        msd2_matched = secondjet.msdcorr * self._msdSF[self._year] * (genflavor2 > 0) + secondjet.msdcorr * (genflavor2 == 0)

        regions = {
            'signal': ['trigger','lumimask','jet1kin','jet2kin','jetid','jetacceptance','n2ddt','antiak4btagMediumOppHem','met','noleptons'],
            'muoncontrol': ['muontrigger', 'minjetkin_muoncr', 'jetid', 'n2ddt', 'ak4btagMedium08', 'noetau', 'onemuon', 'muonDphiAK8'],
            'noselection': [],
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
                for i, cut in enumerate(cuts+['ddbpass']):
                    allcuts.add(cut)
                    cut = selection.all(*allcuts)
                    output['cutflow'].fill(dataset=dataset, region=region, genflavor=normalize(genflavor, cut),
                                        cut=i + 1, weight=weights.weight()[cut])#, msd=normalize(msd_matched, cut))

        if shift_name is None:
            systematics = [
                None,
                'jet_triggerUp',
                'jet_triggerDown',
                'btagWeightUp',
                'btagWeightDown',
                'btagEffStatUp',
                'btagEffStatDown',
            ]
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
            
            output['templates'].fill(
                dataset=dataset,
                region=region,
                systematic=sname,
                pt1=normalize(candidatejet.pt, cut),
                msd1=normalize(msd_matched, cut),
                msd2=normalize(msd2_matched, cut),
                ddb1=normalize(candidatejet.btagDDBvL, cut),
                weight=weight,
            )

            if sname == "nominal":
                output['templates-vh-1'].fill(
                    dataset=dataset,
                    region=region,
                    msd1=normalize(msd_matched, cut),
                    ddb1=normalize(candidatejet.btagDDBvL, cut),
                    pt1=normalize(candidatejet.pt, cut),
                    msd2=normalize(msd2_matched, cut),
                    ddb2=normalize(secondjet.btagDDBvL, cut),
                    pt2=normalize(secondjet.pt, cut),
#                    DR=normalize(DR, cut),
                    weight=weight,
                )

            if not isRealData:
                if wmod is not None:
                    _custom_weight = events.genWeight[cut] * wmod[cut]
                else:
                    _custom_weight = np.ones_like(weight)

        for region in regions:
            cut = selection.all(*(set(regions[region]) - {'n2ddt'}))

            for systematic in systematics:
                if isRealData and systematic is not None:
                    continue
                fill(region, systematic)

        return output

    def postprocess(self, accumulator):
        return accumulator
